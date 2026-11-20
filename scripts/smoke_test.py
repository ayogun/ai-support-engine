#!/usr/bin/env python3
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from scripts import evaluate, lab
from search_lab import core


def unsigned_status(url):
    request = urllib.request.Request(
        url, data=b'{"question":"SMOKE123"}', headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def run(output):
    session = lab.session_for_state()
    state = lab.load_state(session)
    vectors = session.client("s3vectors")
    bucket, index = state["vector_bucket"], state["index"]
    evaluate.check_index(vectors, state, index, core.DIMENSIONS)
    checks = {}
    status = unsigned_status(state["outputs"]["SearchUrl"])
    checks["anonymous_request"] = {"http_status": status, "passed": status == 403}
    status, payload = lab.request_api(session, state, "", "keyword", 3)
    checks["invalid_request"] = {"http_status": status, "passed": status == 400}

    existing = evaluate.index_keys(vectors, bucket, index)
    placeholder_key = None
    try:
        if not existing:
            source_key = f"documents/smoke-{uuid.uuid4().hex}.txt"
            placeholder = core.vector_record(source_key, "SMOKE123 is a temporary keyword-only test record.", [1.0] + [0.0] * (core.DIMENSIONS - 1))
            placeholder_key = placeholder["key"]
            vectors.put_vectors(vectorBucketName=bucket, indexName=index, vectors=[placeholder])
        question = "SMOKE123" if placeholder_key else "AUTH482"
        expected_id = source_key.rsplit("/", 1)[-1].removesuffix(".txt") if placeholder_key else "auth482"
        deadline = time.monotonic() + 60
        while True:
            status, payload = lab.request_api(session, state, question, "keyword", 3)
            identifiers = [item.get("document_id") for item in payload.get("results", [])]
            found = expected_id in identifiers
            if status != 200 or found or time.monotonic() >= deadline:
                break
            time.sleep(2)
        checks["signed_keyword"] = {
            "http_status": status,
            "passed": status == 200 and found,
            "returned_documents": len(identifiers),
            "returned_document_ids": identifiers,
            "temporary_non_model_vector_used_for_keyword_only": bool(placeholder_key),
        }
        status, payload = lab.request_api(session, state, expected_id, "exact", 1)
        exact_ids = [item.get("document_id") for item in payload.get("results", [])]
        checks["signed_exact"] = {"http_status": status, "passed": status == 200 and exact_ids == [expected_id],
                                  "returned_document_ids": exact_ids}
        status, payload = lab.request_api(session, state, expected_id.upper(), "exact", 1)
        checks["signed_exact_case_sensitive"] = {"http_status": status, "passed": status == 200 and payload.get("results") == []}
        missing_id = f"missing-{uuid.uuid4().hex}"
        status, payload = lab.request_api(session, state, missing_id, "exact", 1)
        checks["signed_exact_missing"] = {"http_status": status, "passed": status == 200 and payload.get("results") == []}
        status, payload = lab.request_api(session, state, "../invalid", "exact", 1)
        checks["signed_exact_invalid"] = {"http_status": status, "passed": status == 400}
    finally:
        if placeholder_key:
            vectors.delete_vectors(vectorBucketName=bucket, indexName=index, keys=[placeholder_key])

    for mode in ("vector", "hybrid"):
        status, payload = lab.request_api(session, state, "Why does my session expire?", mode, 3)
        returned = payload.get("results", [])
        checks[f"sagemaker_{mode}"] = {
            "http_status": status, "passed": status == 200 and bool(returned),
            "dependency_code": payload.get("code"), "returned_documents": len(returned),
            "returned_document_ids": [item.get("document_id") for item in returned],
        }
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "region": state["region"], "indexed_vectors_before_checks": len(existing), "checks": checks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    infrastructure_ok = all(checks[name]["passed"] for name in (
        "anonymous_request", "invalid_request", "signed_keyword", "signed_exact", "signed_exact_case_sensitive",
        "signed_exact_missing", "signed_exact_invalid",
    ))
    if not infrastructure_ok:
        return 1
    return 0 if all(checks[f"sagemaker_{mode}"]["passed"] for mode in ("vector", "hybrid")) else 2


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/live-smoke.json")
    try:
        return run(output)
    except (ClientError, BotoCoreError, RuntimeError, ValueError, OSError) as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
