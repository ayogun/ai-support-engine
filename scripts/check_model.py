#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from scripts import lab
from search_lab import core


def run(output):
    session = lab.session_for_state()
    state = lab.load_state(session)
    runtime = session.client("sagemaker-runtime", config=Config(read_timeout=180, retries={"total_max_attempts": 1}))
    observations = []
    for input_type in ("document", "query"):
        started = time.monotonic()
        vector = core.embed_text(
            "A synthetic login session expires after eight hours.",
            state["outputs"]["EmbeddingEndpointName"], input_type=input_type, runtime=runtime,
        )
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=0.001, abs_tol=0.001):
            raise RuntimeError("The embedding endpoint returned a non-unit vector.")
        observations.append({"input_type": input_type, "dimensions": len(vector), "l2_norm": round(norm, 6), "elapsed_ms": elapsed_ms})
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "region": state["region"], "provider": "sagemaker-serverless", "model": core.MODEL_ID,
        "model_revision": lab.MODEL_REVISION, "serving_image_digest": lab.IMAGE_DIGEST,
        "observations": observations,
        "timing_scope": "Sequential SDK calls from one client; includes network and any endpoint startup, not API Gateway.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Invoke the deployed embedding endpoint with document and query inputs.")
    parser.add_argument("--output", type=Path, default=Path("results/sagemaker-model-check.json"))
    arguments = parser.parse_args()
    try:
        run(arguments.output)
        return 0
    except (ClientError, BotoCoreError, RuntimeError, ValueError, OSError) as error:
        print(f"Model check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
