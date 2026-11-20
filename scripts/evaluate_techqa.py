#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from scripts import evaluate, evaluate_scifact, evaluate_support, lab
from search_lab import core


DATASET = "nvidia/TechQA-RAG-Eval"
REVISION = "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a"
FILES = {
    "train.json": ("69d97231509482ed6bd5ec1c4bc0607acb82a88d11169eb8383592d0ca8b93c7", 10 * 1024 * 1024),
    "corpus.zip": ("c06aa287dcc1abf8db6b49b8495df095db73342d729f6451ac330785245d10be", 100 * 1024 * 1024),
}
INDEX_NAME = "techqa-support"
TOP_K = 3


def dataset_path(directory, filename):
    checksum, size_limit = FILES[filename]
    path = Path(directory) / filename
    if path.exists():
        if lab.sha256_file(path) != checksum:
            raise RuntimeError(f"The cached TechQA {filename} does not match the pinned revision.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            if urllib.parse.urlparse(response.geturl()).scheme != "https":
                raise RuntimeError("The TechQA download redirected to an insecure URL.")
            size = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > size_limit:
                    raise RuntimeError(f"The TechQA {filename} download exceeds its size limit.")
                output.write(block)
        if lab.sha256_file(temporary) != checksum:
            raise RuntimeError(f"The downloaded TechQA {filename} does not match the pinned revision.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_dataset(directory):
    rows = json.loads(dataset_path(directory, "train.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("TechQA questions must be a list.")
    documents = {}
    source_count = 0
    with zipfile.ZipFile(dataset_path(directory, "corpus.zip")) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            match = re.fullmatch(r"corpus/([A-Za-z0-9_.-]+)\.txt", member.filename)
            if not match or member.flag_bits & 1:
                raise ValueError("Unexpected or encrypted TechQA archive member.")
            source_count += 1
            if source_count > 50000:
                raise ValueError("The TechQA archive contains too many documents.")
            identifier = match.group(1)
            if identifier in documents:
                raise ValueError("Duplicate TechQA document ID.")
            if member.file_size > core.MAX_DOCUMENT_BYTES - 40:
                continue
            text = archive.read(member).decode("utf-8").strip()
            if text:
                documents[identifier] = text
    return documents, rows, source_count


def select_dataset(documents, rows, source_count, query_count=100, document_count=600):
    candidates = []
    identifiers = set()
    answerable = 0
    unanswerable = 0
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.startswith("DEV_"):
            continue
        if identifier in identifiers:
            raise ValueError("Duplicate TechQA development question ID.")
        identifiers.add(identifier)
        impossible = row.get("is_impossible")
        if type(impossible) is not bool:
            raise ValueError("TechQA answerability must be a boolean.")
        if impossible:
            unanswerable += 1
            continue
        answerable += 1
        contexts = row.get("contexts")
        question = row.get("question")
        if not isinstance(contexts, list) or len(contexts) != 1 or not isinstance(question, str):
            continue
        context = contexts[0]
        filename = context.get("filename", "")
        if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.txt", filename):
            continue
        relevant = filename.removesuffix(".txt")
        question = question.strip()
        if (relevant in documents and isinstance(context.get("text"), str)
                and context["text"].strip() == documents[relevant]
                and question and len(question.encode("utf-8")) <= core.MAX_QUERY_BYTES):
            candidates.append({"id": identifier, "question": question, "relevant": [relevant]})
    if query_count < 1:
        raise ValueError("Query count must be positive.")
    chosen = []
    required = set()
    for query in evaluate_scifact.stable_order(candidates, "techqa-questions", lambda item: item["id"]):
        relevant = query["relevant"][0]
        if relevant not in required:
            chosen.append(query)
            required.add(relevant)
        if len(chosen) == query_count:
            break
    if len(chosen) != query_count:
        raise ValueError("Not enough eligible questions with distinct linked documents.")
    if document_count < len(required) or document_count > len(documents):
        raise ValueError("The document pool cannot contain the required linked documents.")
    remaining = evaluate_scifact.stable_order(set(documents) - required, "techqa-documents", lambda item: item)
    selected_ids = sorted(required | set(remaining[:document_count - len(required)]))
    selected_documents = []
    readiness = {}
    for position, identifier in enumerate(selected_ids):
        marker = f"zztechqaready{position:05d}"
        selected_documents.append((f"techqa/{identifier}.txt", f"{documents[identifier]}\nReadiness marker: {marker}."))
        readiness[identifier] = marker
    counts = {
        "source_documents": source_count, "eligible_documents": len(documents),
        "development_answerable": answerable, "development_unanswerable_excluded": unanswerable,
        "eligible_answerable_questions": len(candidates), "selected_linked_documents": len(required),
    }
    return selected_documents, chosen, readiness, counts


def run(arguments):
    documents, rows, source_count = load_dataset(arguments.dataset)
    documents, queries, readiness, counts = select_dataset(documents, rows, source_count, arguments.queries, arguments.documents)
    session = lab.session_for_state()
    state = lab.load_state(session)
    vectors = session.client("s3vectors")
    runtime = session.client("sagemaker-runtime", config=Config(read_timeout=90, retries={"total_max_attempts": 1}))
    evaluate_scifact.prepare_index(vectors, state, documents, runtime, INDEX_NAME, "TechQA support")
    evaluate_support.wait_for_text_search(vectors, state, readiness, timeout=600, index_name=INDEX_NAME)
    observations = []
    for query in queries:
        embedding = core.embed_text(query["question"], state["outputs"]["EmbeddingEndpointName"],
                                    input_type="query", runtime=runtime)
        for mode in evaluate.MODES:
            request = core.query_request(state["vector_bucket"], INDEX_NAME, query["question"], mode, TOP_K, embedding)
            started = time.perf_counter()
            response = vectors.query_vectors(**request)
            elapsed = (time.perf_counter() - started) * 1000
            returned = [item.get("metadata", {}).get("document_id") for item in response.get("vectors", [])]
            rank = next((position for position, identifier in enumerate(returned, 1)
                         if identifier in query["relevant"]), None)
            observations.append({"query_id": query["id"], "mode": mode, "relevant": query["relevant"],
                                 "returned_ids": returned, "first_relevant_rank": rank, "query_ms": round(elapsed, 2)})
        print("Evaluated TechQA query", query["id"], flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "region": state["region"],
        "dataset": DATASET, "dataset_revision": REVISION,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET}",
        "dataset_license_as_labeled_by_publisher": "Apache-2.0",
        "source_sha256": {name: details[0] for name, details in FILES.items()},
        "selected_data_sha256": evaluate_scifact.corpus_digest(documents, queries),
        "selection": "Fixed SHA-256 order of eligible answerable DEV questions; one question per linked document. Include every linked document, then fixed-hash-sample background documents. Append one unique text-search readiness marker per document.",
        "seed": evaluate_scifact.SEED, "counts": counts, "document_count": len(documents), "query_count": len(queries),
        "top_k": TOP_K, "model": core.MODEL_ID, "model_revision": lab.MODEL_REVISION,
        "model_fingerprint": evaluate_scifact.MODEL_FINGERPRINT,
        "label_definition": "Hit means retrieval of the technical document linked by the original accepted forum answer. Other returned documents have not been adjudicated as irrelevant.",
        "limitations": "Historical IBM forum questions and technical documentation, not the user's support corpus. Answerable questions only; distinct-positive selection, forced inclusion of linked documents, and hash-sampled background make this a constructed pool. Text and query size filtering can bias the sample. The embedding model truncates to 512 tokens, while text search indexes the full stored document. Linked-answer labels are not new independent relevance judgments; other documents may be relevant. No test of abstention, answer generation, or production performance.",
        "timing_scope": "Sequential direct S3 Vectors calls; excludes embedding, API Gateway, ingestion, and readiness checks.",
        "summary": evaluate_scifact.summarize(observations), "observations": observations,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"counts": counts, "summary": report["summary"]}, indent=2))
    print("Wrote", output)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval of original linked TechQA support documents.")
    parser.add_argument("--dataset", default=".build/techqa")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--documents", type=int, default=600)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
        return 0
    except (BotoCoreError, ClientError, RuntimeError, ValueError, OSError, zipfile.BadZipFile) as error:
        print(f"TechQA evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
