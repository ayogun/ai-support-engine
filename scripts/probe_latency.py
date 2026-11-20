#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from scripts import lab


QUESTION = "Find RETURN301"
EXPECTED_DOCUMENT_ID = "return301"
MODES = ("keyword", "vector", "hybrid")


def timed_request(session, state, mode):
    started = time.perf_counter()
    try:
        status, payload = lab.request_api(session, state, QUESTION, mode, 3)
        if status != 200:
            outcome = "http_error"
        elif (not isinstance(payload, dict) or payload.get("mode") != mode
              or not isinstance(payload.get("results"), list)
              or any(not isinstance(item, dict) or not isinstance(item.get("document_id"), str)
                     for item in payload["results"])):
            outcome = "invalid_response"
        else:
            outcome = "success"
        result = {"outcome": outcome, "http_status": status}
        if outcome == "success":
            identifiers = [item["document_id"] for item in payload["results"]]
            result["returned_ids"] = identifiers
            result["first_relevant_rank"] = next(
                (position for position, identifier in enumerate(identifiers, 1)
                 if identifier == EXPECTED_DOCUMENT_ID), None,
            )
        code = payload.get("code") if isinstance(payload, dict) else None
        if outcome != "success" and isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,80}", code):
            result["dependency_code"] = code
    except (BotoCoreError, ClientError, OSError, ValueError) as error:
        result = {"outcome": "client_error", "error_type": type(error).__name__}
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    result["_started"] = started
    return result


def summarize(samples):
    successes = sorted(item["elapsed_ms"] for item in samples if item["outcome"] == "success")
    responses = sorted(item["elapsed_ms"] for item in samples if "http_status" in item)
    judged = [item for item in samples if item["outcome"] == "success" and "first_relevant_rank" in item]
    outcomes = Counter(item["outcome"] for item in samples)
    statuses = Counter(str(item["http_status"]) for item in samples if "http_status" in item)
    return {
        "attempts": len(samples), "successes": len(successes), "outcomes": dict(sorted(outcomes.items())),
        "http_statuses": dict(sorted(statuses.items())),
        "http_responses": len(responses),
        "http_response_p50_ms": round(statistics.median(responses), 1) if responses else None,
        "http_response_p95_ms": round(responses[math.ceil(0.95 * len(responses)) - 1], 1) if responses else None,
        "retrieval_evaluated": len(judged),
        "retrieval_hit_at_1": sum(item["first_relevant_rank"] == 1 for item in judged),
        "retrieval_hit_at_3": sum(item["first_relevant_rank"] is not None and item["first_relevant_rank"] <= 3 for item in judged),
        "retrieval_misses": sum(item["first_relevant_rank"] is None for item in judged),
        "success_p50_ms": round(statistics.median(successes), 1) if successes else None,
        "success_p95_ms": round(successes[math.ceil(0.95 * len(successes)) - 1], 1) if successes else None,
    }


def burst(state, concurrency):
    barrier = threading.Barrier(concurrency, timeout=15)

    def worker():
        try:
            session = boto3.Session(region_name=state["region"])
            session.get_credentials().get_frozen_credentials()
            barrier.wait()
            return timed_request(session, state, "hybrid")
        except (BotoCoreError, ClientError, OSError, AttributeError, threading.BrokenBarrierError) as error:
            barrier.abort()
            return {"outcome": "client_error", "error_type": type(error).__name__, "elapsed_ms": 0.0}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(executor.map(lambda unused: worker(), range(concurrency)))
    starts = [item.pop("_started") for item in samples if "_started" in item]
    return {"client_start_skew_ms": round((max(starts) - min(starts)) * 1000, 2) if starts else None,
            "summary": summarize(samples), "samples": samples}


def run(arguments):
    session = lab.session_for_state()
    state = lab.load_state(session)
    warmups = {}
    sequential = {}
    for mode in MODES:
        warmup = timed_request(session, state, mode)
        warmup.pop("_started")
        warmups[mode] = warmup
        samples = []
        for unused in range(arguments.samples):
            observation = timed_request(session, state, mode)
            observation.pop("_started")
            samples.append(observation)
        sequential[mode] = {"summary": summarize(samples), "samples": samples}
        print("Completed sequential", mode, "requests.", flush=True)
    bursts = []
    for number in range(arguments.burst_rounds):
        if number:
            time.sleep(2)
        bursts.append(burst(state, arguments.concurrency))
        print("Completed hybrid burst", number + 1, flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "region": state["region"],
        "question": QUESTION, "expected_document_id": EXPECTED_DOCUMENT_ID, "top_k": 3,
        "scope": "Client-side signed HTTP round trips include signing, network, API Gateway, Lambda, S3 Vectors, and model inference for vector/hybrid. Warm-up requests are reported separately.",
        "limitations": "Small repeated-query sample, fixed mode order, one client, nearest-rank p95; not a capacity test or latency SLA. HTTP success and retrieval correctness are distinct. Failed requests are excluded from success percentiles and counted separately.",
        "warmups": warmups, "sequential": sequential,
        "burst_concurrency": arguments.concurrency, "bursts": bursts,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({mode: value["summary"] for mode, value in sequential.items()}, indent=2))
    print("Burst outcomes:", [item["summary"] for item in bursts])
    print("Wrote", output)


def main():
    parser = argparse.ArgumentParser(description="Measure bounded signed HTTP search requests, including a small hybrid burst.")
    parser.add_argument("--samples", type=int, default=10, choices=range(1, 31), metavar="1-30")
    parser.add_argument("--concurrency", type=int, default=3, choices=range(1, 4), metavar="1-3")
    parser.add_argument("--burst-rounds", type=int, default=3, choices=range(1, 6), metavar="1-5")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
        return 0
    except (BotoCoreError, ClientError, RuntimeError, OSError) as error:
        print(f"Latency probe failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
