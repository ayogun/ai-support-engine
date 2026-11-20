#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from scripts import evaluate, lab
from search_lab import core


DATASET_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATASET_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
DATASET_LIMIT = 10 * 1024 * 1024
SEED = "20261118"
INDEX_NAME = "scifact-evaluation"
LEXICAL_INDEX_NAME = "scifact-lexical"
TOP_K = 3
MODEL_FINGERPRINT = hashlib.sha256(
    (lab.MODEL_REVISION + ":" + lab.IMAGE_DIGEST + ":" + lab.sha256_file(lab.ROOT / "src" / "model" / "inference.py")).encode()
).hexdigest()


def archive_path(path):
    path = Path(path)
    if path.exists():
        if lab.sha256_file(path) != DATASET_SHA256:
            raise RuntimeError("The SciFact archive checksum does not match the pinned release.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=120) as response, temporary.open("wb") as output:
            if urllib.parse.urlparse(response.geturl()).scheme != "https":
                raise RuntimeError("The SciFact download redirected to an insecure URL.")
            size = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > DATASET_LIMIT:
                    raise RuntimeError("The SciFact archive exceeds the download size limit.")
                output.write(block)
        if lab.sha256_file(temporary) != DATASET_SHA256:
            raise RuntimeError("The downloaded SciFact archive is not the pinned release.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_dataset(path):
    with tarfile.open(archive_path(path), "r:gz") as archive:
        records = []
        for name in ("data/corpus.jsonl", "data/claims_dev.jsonl"):
            member = archive.getmember(name)
            if not member.isfile() or member.size > 25 * 1024 * 1024:
                raise RuntimeError(f"Unexpected SciFact archive member: {name}.")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Could not read SciFact archive member: {name}.")
            with stream:
                records.append([json.loads(line) for line in stream])
    return records


def stable_order(values, namespace, key):
    def sort_key(value):
        identifier = str(key(value))
        digest = hashlib.sha256(f"{SEED}:{namespace}:{identifier}".encode()).digest()
        return digest, identifier

    return sorted(values, key=sort_key)


def lexical_pool(documents, queries, required, corpus_count):
    postings = defaultdict(list)
    document_lengths = {}
    for identifier, text in documents.items():
        terms = re.findall(r"[a-z0-9]+", text.lower())
        document_lengths[identifier] = len(terms)
        for term, frequency in Counter(terms).items():
            postings[term].append((identifier, frequency))
    average_length = sum(document_lengths.values()) / len(document_lengths)
    rankings = []
    for query in queries:
        scores = defaultdict(float)
        for term in sorted(set(re.findall(r"[a-z0-9]+", query["question"].lower()))):
            matches = postings.get(term, [])
            if not matches:
                continue
            inverse_frequency = math.log(1 + (len(documents) - len(matches) + 0.5) / (len(matches) + 0.5))
            for identifier, frequency in matches:
                if identifier in required:
                    continue
                length = document_lengths[identifier]
                denominator = frequency + 1.2 * (0.25 + 0.75 * length / average_length)
                scores[identifier] += inverse_frequency * frequency * 2.2 / denominator
        rankings.append([identifier for identifier, unused in sorted(scores.items(), key=lambda item: (-item[1], item[0]))])
    selected = set(required)
    offsets = [0] * len(rankings)
    while len(selected) < corpus_count:
        added = False
        for position, ranking in enumerate(rankings):
            while offsets[position] < len(ranking) and ranking[offsets[position]] in selected:
                offsets[position] += 1
            if offsets[position] < len(ranking):
                selected.add(ranking[offsets[position]])
                offsets[position] += 1
                added = True
                if len(selected) == corpus_count:
                    break
        if not added:
            break
    remaining = stable_order(set(documents) - selected, "documents", lambda identifier: identifier)
    selected.update(remaining[:corpus_count - len(selected)])
    return selected


def select_dataset(corpus, claims, query_count=50, corpus_count=300, pool="random"):
    documents = {}
    for item in corpus:
        identifier = str(item["doc_id"])
        text = item["title"].strip() + "\n\n" + " ".join(item["abstract"]).strip()
        if identifier in documents:
            raise RuntimeError("Duplicate SciFact document ID.")
        if text.strip() and len(text.encode("utf-8")) <= core.MAX_DOCUMENT_BYTES:
            documents[identifier] = text

    candidates = []
    seen_claims = set()
    for item in claims:
        identifier = str(item["id"])
        if identifier in seen_claims:
            raise RuntimeError("Duplicate SciFact claim ID.")
        seen_claims.add(identifier)
        evidence = item.get("evidence", {})
        relevant = sorted(str(document_id) for document_id, annotations in evidence.items()
                          if annotations and all(note.get("label") in {"SUPPORT", "CONTRADICT"} for note in annotations))
        question = item["claim"].strip()
        if (relevant and len(relevant) == len(evidence) and all(identifier in documents for identifier in relevant)
                and question and len(question.encode("utf-8")) <= core.MAX_QUERY_BYTES):
            candidates.append({"id": identifier, "question": question, "relevant": relevant})

    if query_count < 1 or query_count > len(candidates):
        raise ValueError("Requested claim count exceeds eligible evidence-bearing development claims.")
    selected = stable_order(candidates, "claims", lambda item: item["id"])[:query_count]
    required = {identifier for item in selected for identifier in item["relevant"]}
    if corpus_count < len(required) or corpus_count > len(documents):
        raise ValueError("Corpus size cannot include the selected claims' evidence documents.")
    if pool == "random":
        remaining = stable_order(set(documents) - required, "documents", lambda identifier: identifier)
        selected_ids = sorted(required | set(remaining[:corpus_count - len(required)]))
    elif pool == "lexical":
        selected_ids = sorted(lexical_pool(documents, selected, required, corpus_count))
    else:
        raise ValueError("Pool must be random or lexical.")
    selected_documents = [(f"scifact/{identifier}.txt", documents[identifier]) for identifier in selected_ids]
    counts = {
        "source_documents": len(corpus),
        "eligible_documents": len(documents),
        "development_claims": len(claims),
        "eligible_evidence_claims": len(candidates),
        "selected_evidence_documents": len(required),
    }
    return selected_documents, selected, counts


def corpus_digest(documents, queries):
    digest = hashlib.sha256()
    for key, text in documents:
        digest.update(key.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    digest.update(json.dumps(queries, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def source_version(text):
    return MODEL_FINGERPRINT + ":" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_index(vectors, state, documents, runtime, index_name=INDEX_NAME, dataset_label="SciFact"):
    if index_name not in state["indexes"]:
        state["indexes"].append(index_name)
        lab.write_state(state)
    try:
        evaluate.check_index(vectors, state, index_name, core.DIMENSIONS)
    except ClientError as error:
        if lab.error_code(error) not in {"NotFoundException", "ResourceNotFoundException"}:
            raise
        lab.create_index(vectors, state, index_name, core.DIMENSIONS)
        evaluate.check_index(vectors, state, index_name, core.DIMENSIONS)

    expected = {core.vector_key(key): (key, text) for key, text in documents}
    existing = evaluate.index_keys(vectors, state["vector_bucket"], index_name)
    if existing - expected.keys():
        raise RuntimeError(f"The {dataset_label} index contains unexpected vectors; refusing to overwrite it.")
    existing_keys = sorted(existing)
    for start in range(0, len(existing_keys), 100):
        batch = existing_keys[start:start + 100]
        response = vectors.get_vectors(vectorBucketName=state["vector_bucket"], indexName=index_name,
                                       keys=batch, returnMetadata=True)
        returned = {item["key"]: item for item in response.get("vectors", [])}
        if set(returned) != set(batch):
            raise RuntimeError(f"Could not verify the existing {dataset_label} vectors.")
        for key in batch:
            if returned[key].get("metadata", {}).get("source_version") != source_version(expected[key][1]):
                raise RuntimeError(f"An existing {dataset_label} vector has different source text or model revision.")

    pending = [(key, text) for key, text in documents if core.vector_key(key) not in existing]
    for start in range(0, len(pending), 25):
        batch = pending[start:start + 25]
        records = []
        for key, text in batch:
            embedding = core.embed_text(text, state["outputs"]["EmbeddingEndpointName"],
                                        input_type="document", runtime=runtime)
            records.append(core.vector_record(key, text, embedding, source_version(text)))
        vectors.put_vectors(vectorBucketName=state["vector_bucket"], indexName=index_name, vectors=records)
        print(f"Indexed {len(existing) + min(start + len(batch), len(pending))}/{len(documents)} {dataset_label} documents.", flush=True)

    evaluate.wait_for_corpus(vectors, state["vector_bucket"], index_name, documents, timeout=120)
    deadline = time.monotonic() + 120
    while True:
        index = evaluate.check_index(vectors, state, index_name, core.DIMENSIONS)
        fields = index["metadataConfiguration"]["textSearchConfigurations"]
        if any(field.get("metadataKey") == "text" and field.get("status") == "ACTIVE" for field in fields):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"The {dataset_label} text-search configuration did not become ACTIVE.")
        time.sleep(3)


def summarize(observations):
    summary = {}
    for mode in evaluate.MODES:
        selected = [item for item in observations if item["mode"] == mode]
        if not selected:
            continue
        ranks = [item["first_relevant_rank"] for item in selected]
        recalls = [len(set(item["returned_ids"]) & set(item["relevant"])) / len(item["relevant"]) for item in selected]
        durations = sorted(item["query_ms"] for item in selected)
        summary[mode] = {
            "queries": len(selected),
            "hit_at_1": sum(rank == 1 for rank in ranks),
            "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks),
            "mrr_at_3": round(sum(1 / rank if rank and rank <= 3 else 0 for rank in ranks) / len(ranks), 4),
            "macro_recall_at_3": round(sum(recalls) / len(recalls), 4),
            "query_ms_median": round(statistics.median(durations), 1),
            "query_ms_p95": round(durations[math.ceil(0.95 * len(durations)) - 1], 1),
        }
    return summary


def run(arguments):
    corpus, claims = load_dataset(arguments.archive)
    documents, queries, counts = select_dataset(corpus, claims, arguments.queries, arguments.documents, arguments.pool)
    index_name = INDEX_NAME if arguments.pool == "random" else LEXICAL_INDEX_NAME
    session = lab.session_for_state()
    state = lab.load_state(session)
    vectors = session.client("s3vectors")
    runtime = session.client("sagemaker-runtime", config=Config(read_timeout=90, retries={"total_max_attempts": 1}))
    prepare_index(vectors, state, documents, runtime, index_name)
    observations = []
    for query in queries:
        embedding = core.embed_text(query["question"], state["outputs"]["EmbeddingEndpointName"],
                                    input_type="query", runtime=runtime)
        for mode in evaluate.MODES:
            request = core.query_request(state["vector_bucket"], index_name, query["question"], mode, TOP_K, embedding)
            started = time.perf_counter()
            response = vectors.query_vectors(**request)
            elapsed = (time.perf_counter() - started) * 1000
            returned = response.get("vectors", [])
            identifiers = [item.get("metadata", {}).get("document_id") for item in returned]
            rank = next((position for position, identifier in enumerate(identifiers, 1)
                         if identifier in query["relevant"]), None)
            observations.append({
                "claim_id": query["id"], "mode": mode, "relevant": query["relevant"],
                "returned_ids": identifiers, "first_relevant_rank": rank,
                "query_ms": round(elapsed, 2),
                "scores": [item.get("scores") for item in returned],
            })
        print("Evaluated SciFact claim", query["id"], flush=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "region": state["region"], "dataset": "SciFact original release, development split",
        "dataset_url": DATASET_URL, "dataset_sha256": DATASET_SHA256,
        "seed": SEED, "pool": arguments.pool,
        "selection": ("SHA-256 claim order; include all annotated evidence, then SHA-256-order background documents"
                      if arguments.pool == "random" else
                      "SHA-256 claim order; include all annotated evidence, then round-robin high-BM25 lexical-overlap documents from the full corpus"),
        "selected_data_sha256": corpus_digest(documents, queries), "source_counts": counts,
        "document_count": len(documents), "query_count": len(queries), "top_k": TOP_K,
        "selected_document_ids": [Path(key).stem for key, _ in documents],
        "selected_claims": queries, "model": core.MODEL_ID, "model_revision": lab.MODEL_REVISION,
        "model_fingerprint": MODEL_FINGERPRINT,
        "timing_scope": "Sequential direct S3 Vectors calls; excludes model inference and API Gateway.",
        "label_limitations": "Only annotated SUPPORT/CONTRADICT evidence counts; other documents may be relevant but unjudged. Development claims without evidence are excluded.",
        "summary": summarize(observations), "observations": observations,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print("Wrote", output)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a bounded SciFact development subset with SageMaker and S3 Vectors.")
    parser.add_argument("--archive", default=".build/scifact-data.tar.gz")
    parser.add_argument("--queries", type=int, default=50, choices=range(1, 101), metavar="1-100")
    parser.add_argument("--documents", type=int, default=300, choices=range(1, 501), metavar="1-500")
    parser.add_argument("--pool", choices=["random", "lexical"], default="random")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
        return 0
    except (BotoCoreError, ClientError, RuntimeError, ValueError, OSError, tarfile.TarError) as error:
        print(f"SciFact evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
