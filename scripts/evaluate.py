#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from scripts import lab
from search_lab import core


ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_INDEX = "bge-evaluation"
LOCAL_DIMENSIONS = 384
MODES = ("keyword", "vector", "hybrid")


def fixtures():
    paths = sorted((ROOT / "fixtures" / "documents").glob("*.txt"))
    queries_path = ROOT / "fixtures" / "queries.json"
    documents = [(f"documents/{path.name}", path.read_text()) for path in paths]
    queries = json.loads(queries_path.read_text())
    if not documents or not queries:
        raise RuntimeError("The document and query fixtures must not be empty.")
    document_ids = {Path(key).stem for key, _ in documents}
    query_ids = set()
    for query in queries:
        if (not isinstance(query, dict) or not isinstance(query.get("id"), str)
                or query["id"] in query_ids or query.get("kind") not in {"exact_identifier", "paraphrase"}
                or not isinstance(query.get("question"), str) or not query["question"].strip()
                or not isinstance(query.get("relevant"), list) or not query["relevant"]
                or any(identifier not in document_ids for identifier in query["relevant"])):
            raise RuntimeError("Invalid, duplicate, or unresolvable query fixture.")
        query_ids.add(query["id"])
    checksum = hashlib.sha256()
    for path in [*paths, queries_path]:
        checksum.update(str(path.relative_to(ROOT)).encode())
        checksum.update(b"\0")
        checksum.update(path.read_bytes())
        checksum.update(b"\0")
    return documents, queries, checksum.hexdigest()


def index_keys(vectors, bucket, index):
    paginator = vectors.get_paginator("list_vectors")
    return {
        entry["key"]
        for page in paginator.paginate(vectorBucketName=bucket, indexName=index)
        for entry in page.get("vectors", [])
    }


def check_index(vectors, state, index_name, dimensions):
    index = vectors.get_index(vectorBucketName=state["vector_bucket"], indexName=index_name)["index"]
    tags = vectors.list_tags_for_resource(resourceArn=index["indexArn"])["tags"]
    if not lab.has_tags(tags, state["lab_id"]):
        raise RuntimeError("The evaluation index does not have this lab's ownership tags.")
    text_fields = index.get("metadataConfiguration", {}).get("textSearchConfigurations", [])
    if index.get("dimension") != dimensions or index.get("indexMode") != "ENHANCED" or not any(field.get("metadataKey") == "text" for field in text_fields):
        raise RuntimeError("The index has incompatible dimensions, mode, or text-search configuration.")
    return index


def prepare_local_index(vectors, state, documents, embeddings):
    try:
        check_index(vectors, state, LOCAL_INDEX, LOCAL_DIMENSIONS)
    except ClientError as error:
        if lab.error_code(error) not in {"NotFoundException", "ResourceNotFoundException"}:
            raise
        lab.create_index(vectors, state, LOCAL_INDEX, LOCAL_DIMENSIONS)
        check_index(vectors, state, LOCAL_INDEX, LOCAL_DIMENSIONS)
    expected = {core.vector_key(key) for key, _ in documents}
    if index_keys(vectors, state["vector_bucket"], LOCAL_INDEX) - expected:
        raise RuntimeError("The local evaluation index has unexpected vectors; refusing to overwrite it.")
    records = [core.vector_record(key, text, embedding.tolist()) for (key, text), embedding in zip(documents, embeddings, strict=True)]
    vectors.put_vectors(vectorBucketName=state["vector_bucket"], indexName=LOCAL_INDEX, vectors=records)
    return LOCAL_INDEX


def wait_for_corpus(vectors, bucket, index, documents, timeout=90):
    expected = {core.vector_key(key) for key, _ in documents}
    deadline = time.monotonic() + timeout
    while True:
        actual = index_keys(vectors, bucket, index)
        if actual == expected:
            return
        if actual - expected:
            raise RuntimeError("The index contains unexpected vectors; refusing an invalid comparison.")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Only {len(actual)} of {len(expected)} expected vectors are visible; check ingestion and retry.")
        time.sleep(3)


def wait_for_text_search(vectors, bucket, index, documents, timeout=90):
    deadline = time.monotonic() + timeout
    pending = {Path(key).stem for key, _ in documents}
    while pending:
        for document_id in sorted(pending.copy()):
            request = core.query_request(bucket, index, document_id.upper(), "keyword", 3)
            response = vectors.query_vectors(**request)
            if any(item.get("metadata", {}).get("document_id") == document_id for item in response.get("vectors", [])):
                pending.remove(document_id)
        if pending and time.monotonic() >= deadline:
            raise RuntimeError(f"Text search is not ready for {len(pending)} fixture documents.")
        if pending:
            time.sleep(3)


def summarize(observations):
    summaries = {}
    for group in ("all", "exact_identifier", "paraphrase"):
        summaries[group] = {}
        for mode in MODES:
            selected = [item for item in observations if item["mode"] == mode and (group == "all" or item["kind"] == group)]
            if not selected:
                continue
            ranks = [item["rank"] for item in selected]
            elapsed = sorted(item["query_ms"] for item in selected)
            summaries[group][mode] = {
                "queries": len(selected),
                "hit_at_1": sum(rank == 1 for rank in ranks),
                "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks),
                "mrr_at_3": round(sum(1 / rank if rank and rank <= 3 else 0 for rank in ranks) / len(ranks), 4),
                "query_ms_median": round(statistics.median(elapsed), 1),
                "query_ms_p95": round(elapsed[math.ceil(0.95 * len(elapsed)) - 1], 1),
            }
    return summaries


def evaluate(arguments):
    session = lab.session_for_state()
    state = lab.load_state(session)
    vectors = session.client("s3vectors")
    documents, queries, fixture_sha256 = fixtures()
    if arguments.embedding == "bge":
        from fastembed import TextEmbedding

        print("Loading local BGE model; this does not test the SageMaker endpoint.", flush=True)
        model = TextEmbedding(model_name=LOCAL_MODEL, threads=2)
        document_embeddings = list(model.passage_embed([text for _, text in documents]))
        index_name = prepare_local_index(vectors, state, documents, document_embeddings)
        model_directory = getattr(model.model, "_model_dir", None)
        model_description = {
            "provider": "local-fastembed", "model": LOCAL_MODEL, "dimensions": LOCAL_DIMENSIONS,
            "fastembed_version": importlib.metadata.version("fastembed"),
            "artifact_snapshot": Path(model_directory).name if model_directory else None,
        }
    else:
        index_name = state["index"]
        check_index(vectors, state, index_name, core.DIMENSIONS)
        runtime = session.client("sagemaker-runtime", config=Config(read_timeout=90, retries={"total_max_attempts": 1}))
        model_description = {
            "provider": "sagemaker-serverless", "model": core.MODEL_ID, "dimensions": core.DIMENSIONS,
            "model_revision": lab.MODEL_REVISION, "serving_image_digest": lab.IMAGE_DIGEST,
        }
    wait_for_corpus(vectors, state["vector_bucket"], index_name, documents)
    wait_for_text_search(vectors, state["vector_bucket"], index_name, documents)
    if arguments.embedding == "bge":
        question_embeddings = [value.tolist() for value in model.query_embed([query["question"] for query in queries])]
    else:
        print("Embedding questions with the SageMaker serverless endpoint...", flush=True)
        question_embeddings = [
            core.embed_text(query["question"], state["outputs"]["EmbeddingEndpointName"], input_type="query", runtime=runtime)
            for query in queries
        ]
    observations = []
    for query, embedding in zip(queries, question_embeddings, strict=True):
        for mode in MODES:
            request = core.query_request(state["vector_bucket"], index_name, query["question"], mode, 3, embedding)
            started = time.perf_counter()
            response = vectors.query_vectors(**request)
            elapsed_ms = (time.perf_counter() - started) * 1000
            returned = response.get("vectors", [])
            document_ids = [entry.get("metadata", {}).get("document_id") for entry in returned]
            rank = next((position for position, identifier in enumerate(document_ids, 1) if identifier in query["relevant"]), None)
            observations.append({
                "query_id": query["id"], "kind": query["kind"], "question": query["question"],
                "relevant": query["relevant"], "mode": mode, "rank": rank,
                "query_ms": round(elapsed_ms, 2), "results": returned,
            })
        print("Evaluated", query["id"], flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "region": state["region"], "index_name": index_name,
        "embedding": model_description, "fixture_sha256": fixture_sha256,
        "document_count": len(documents), "query_count": len(queries), "top_k": 3,
        "timing_scope": "Sequential client-side S3 Vectors query calls; includes network and SDK overhead, excludes embedding and API Gateway.",
        "summary": summarize(observations), "observations": observations,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print("Wrote", output)


def main():
    parser = argparse.ArgumentParser(description="Compare S3 Vectors keyword, vector, and hybrid search on labeled synthetic queries.")
    parser.add_argument("--embedding", choices=["sagemaker", "bge"], default="sagemaker")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        evaluate(arguments)
        return 0
    except (ClientError, BotoCoreError, RuntimeError, ValueError, OSError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
