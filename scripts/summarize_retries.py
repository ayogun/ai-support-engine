#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

from scripts import probe_latency


COMPARABLE_FIELDS = (
    "region", "question", "expected_document_id", "top_k", "endpoint_max_concurrency",
    "concurrency", "rounds_per_block", "pause_between_bursts_seconds", "read_timeout_seconds",
)


def summarize_reports(reports):
    if not reports:
        raise ValueError("At least one retry report is required.")
    reference = {field: reports[0][field] for field in COMPARABLE_FIELDS}
    pooled = {1: [], 3: []}
    runs = []
    for report in reports:
        if {field: report[field] for field in COMPARABLE_FIELDS} != reference:
            raise ValueError("Retry reports do not have matching workload settings.")
        blocks = report["blocks"]
        if [block["model_total_max_attempts"] for block in blocks] != [1, 3, 3, 1]:
            raise ValueError("Retry reports must contain the complete ABBA block sequence.")
        run_samples = {1: [], 3: []}
        for block in blocks:
            if len(block["bursts"]) != reference["rounds_per_block"]:
                raise ValueError("A retry block has a missing or unexpected burst.")
            for burst in block["bursts"]:
                if len(burst["samples"]) != reference["concurrency"]:
                    raise ValueError("A retry burst has a missing or unexpected request.")
                run_samples[block["model_total_max_attempts"]].extend(burst["samples"])
        for attempts, samples in run_samples.items():
            pooled[attempts].extend(samples)
        runs.append({str(attempts): probe_latency.summarize(samples) for attempts, samples in run_samples.items()})
    return {
        "workload": reference, "runs": runs,
        "pooled": {str(attempts): probe_latency.summarize(samples) for attempts, samples in pooled.items()},
        "limitations": "Pooling does not make requests independent or establish production reliability. HTTP-response latency includes errors but excludes requests without an HTTP response; latency of successful requests has a different denominator. Matching workload fields alone cannot prove that all runtime conditions were identical.",
    }


def run(arguments):
    reports = []
    sources = []
    for name in arguments.reports:
        path = Path(name)
        data = path.read_bytes()
        reports.append(json.loads(data))
        sources.append({"report_file": path.name, "report_sha256": hashlib.sha256(data).hexdigest()})
    result = summarize_reports(reports)
    result["sources"] = sources
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["pooled"], indent=2))
    print("Wrote", output)


def main():
    parser = argparse.ArgumentParser(description="Summarize compatible ABBA retry experiments from their individual request observations.")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
        return 0
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"Retry summary failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
