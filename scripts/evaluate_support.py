#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from scripts import evaluate, evaluate_scifact, lab
from search_lab import core


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "support_cases.json"
INDEX_NAME = "support-evaluation"
TOP_K = 3


def load_fixture(path=FIXTURE):
    raw = Path(path).read_bytes()
    fixture = json.loads(raw)
    documents = fixture.get("documents")
    queries = fixture.get("queries")
    missing = fixture.get("missing_identifiers")
    if not isinstance(documents, list) or not documents or not isinstance(queries, list) or not queries or not isinstance(missing, list):
        raise ValueError("The support fixture must contain documents, queries, and missing identifiers.")
    identifiers = set()
    indexed_documents = []
    readiness = {}
    for position, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError("Invalid support document.")
        identifier, text = document.get("id"), document.get("text")
        key = core.identifier_source_key(identifier)
        if identifier in identifiers or not isinstance(text, str) or not text.strip() or identifier not in text:
            raise ValueError("Support document IDs must be unique and appear in their text.")
        marker = f"zzreadymarker{position:04d}"
        content = f"{text}\nReadiness marker: {marker}."
        if len(content.encode("utf-8")) > core.MAX_DOCUMENT_BYTES:
            raise ValueError("A support document exceeds the lab size limit.")
        identifiers.add(identifier)
        indexed_documents.append((key, content))
        readiness[identifier] = marker
    query_ids = set()
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("Invalid support query.")
        identifier = query.get("id")
        relevant = query.get("relevant")
        if (not isinstance(identifier, str) or not identifier or identifier in query_ids
                or query.get("kind") not in {"exact_identifier", "paraphrase"}
                or not isinstance(query.get("question"), str) or not query["question"].strip()
                or not isinstance(relevant, list) or len(relevant) != 1 or relevant[0] not in identifiers):
            raise ValueError("Invalid, duplicate, or unresolvable support query.")
        if query["kind"] == "exact_identifier" and query.get("identifier") != relevant[0]:
            raise ValueError("An identifier query must identify its labeled document exactly.")
        query_ids.add(identifier)
    if len(set(missing)) != len(missing):
        raise ValueError("Missing identifiers must be unique.")
    for identifier in missing:
        core.identifier_source_key(identifier)
        if identifier in identifiers:
            raise ValueError("A missing identifier exists in the support corpus.")
    return indexed_documents, queries, missing, readiness, hashlib.sha256(raw).hexdigest()


def wait_for_text_search(vectors, state, readiness, timeout=120, index_name=INDEX_NAME):
    pending = dict(readiness)
    deadline = time.monotonic() + timeout
    while pending:
        for identifier, marker in list(pending.items()):
            request = core.query_request(state["vector_bucket"], index_name, marker, "keyword", 1)
            response = vectors.query_vectors(**request)
            if any(item.get("metadata", {}).get("document_id") == identifier for item in response.get("vectors", [])):
                del pending[identifier]
        if pending and time.monotonic() >= deadline:
            raise RuntimeError(f"Only {len(readiness) - len(pending)} of {len(readiness)} support documents are text-searchable.")
        if pending:
            time.sleep(3)


def run(arguments):
    documents, queries, missing, readiness, fixture_sha256 = load_fixture()
    session = lab.session_for_state()
    state = lab.load_state(session)
    vectors = session.client("s3vectors")
    runtime = session.client("sagemaker-runtime", config=Config(read_timeout=90, retries={"total_max_attempts": 1}))
    evaluate_scifact.prepare_index(vectors, state, documents, runtime, INDEX_NAME, "synthetic support")
    wait_for_text_search(vectors, state, readiness)
    config = {"vector_bucket": state["vector_bucket"], "index_name": INDEX_NAME}
    observations = []
    exact_checks = []
    for query in queries:
        embedding = core.embed_text(query["question"], state["outputs"]["EmbeddingEndpointName"],
                                    input_type="query", runtime=runtime)
        for mode in evaluate.MODES:
            request = core.query_request(state["vector_bucket"], INDEX_NAME, query["question"], mode, TOP_K, embedding)
            started = time.perf_counter()
            response = vectors.query_vectors(**request)
            elapsed = (time.perf_counter() - started) * 1000
            returned = response.get("vectors", [])
            identifiers = [item.get("metadata", {}).get("document_id") for item in returned]
            rank = next((position for position, identifier in enumerate(identifiers, 1)
                         if identifier in query["relevant"]), None)
            observations.append({
                "query_id": query["id"], "kind": query["kind"], "mode": mode,
                "relevant": query["relevant"], "returned_ids": identifiers, "rank": rank,
                "query_ms": round(elapsed, 2), "scores": [item.get("scores") for item in returned],
            })
        if query["kind"] == "exact_identifier":
            actual = core.exact_lookup(query["identifier"], config, vectors)["results"]
            returned_ids = [result["document_id"] for result in actual]
            exact_checks.append({"query_id": query["id"], "identifier": query["identifier"],
                                 "expected": query["identifier"], "returned_ids": returned_ids,
                                 "correct": returned_ids == [query["identifier"]]})
        print("Evaluated synthetic support query", query["id"], flush=True)
    for identifier in missing:
        actual = core.exact_lookup(identifier, config, vectors)["results"]
        returned_ids = [result["document_id"] for result in actual]
        exact_checks.append({"identifier": identifier, "expected": None, "returned_ids": returned_ids,
                             "correct": returned_ids == []})
    positives = [item for item in exact_checks if item["expected"] is not None]
    negatives = [item for item in exact_checks if item["expected"] is None]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "region": state["region"],
        "dataset": "Author-created synthetic support tickets and questions; not independently labeled",
        "fixture_sha256": fixture_sha256, "document_count": len(documents), "query_count": len(queries),
        "negative_exact_lookups": len(missing), "top_k": TOP_K,
        "model": core.MODEL_ID, "model_revision": lab.MODEL_REVISION,
        "model_fingerprint": evaluate_scifact.MODEL_FINGERPRINT,
        "construction": "Each synthetic document includes a unique readiness marker, absent from queries; vectors were written directly to a separate index.",
        "timing_scope": "Sequential direct S3 Vectors calls; excludes model inference and API Gateway.",
        "limitations": "Small author-written corpus and labels with constructed lookalike IDs; not representative of a production support corpus or independent relevance judgments.",
        "summary": evaluate.summarize(observations),
        "exact_summary": {"positive_correct": sum(item["correct"] for item in positives), "positive_total": len(positives),
                          "negative_correct": sum(item["correct"] for item in negatives), "negative_total": len(negatives)},
        "observations": observations, "exact_checks": exact_checks,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": report["summary"], "exact": report["exact_summary"]}, indent=2))
    print("Wrote", output)


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic support cases and exact identifier lookups.")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
        return 0
    except (BotoCoreError, ClientError, RuntimeError, ValueError, OSError) as error:
        print(f"Support evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
