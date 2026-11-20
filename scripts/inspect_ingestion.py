#!/usr/bin/env python3
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from scripts import lab


def run(output):
    session = lab.session_for_state()
    state = lab.load_state(session)
    logs = session.client("logs")
    group = "/aws/lambda/" + state["outputs"]["IngestFunctionName"]
    paginator = logs.get_paginator("filter_log_events")
    events = [entry for page in paginator.paginate(logGroupName=group) for entry in page.get("events", [])]
    throttles = sum("ThrottlingException" in entry["message"] for entry in events)
    access_denied = sum("AccessDeniedException" in entry["message"] for entry in events)
    oversized = sum("Document exceeds the lab size limit" in entry["message"] for entry in events)
    queue = session.client("sqs")
    queue_url = state["outputs"]["FailedIngestQueueUrl"]
    attributes = queue.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["ApproximateNumberOfMessages"])["Attributes"]
    messages = queue.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1, VisibilityTimeout=0, MessageAttributeNames=["All"]).get("Messages", [])
    keys = []
    for message in messages:
        event = json.loads(message["Body"])
        keys.extend(record.get("s3", {}).get("object", {}).get("key", "") for record in event.get("Records", []))
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "region": state["region"],
        "ingest_throttle_log_events": throttles,
        "ingest_access_denied_log_events": access_denied,
        "ingest_oversized_document_log_events": oversized,
        "approximate_visible_dlq_messages": int(attributes["ApproximateNumberOfMessages"]),
        "sample_event_keys": keys,
        "note": "The queue sample used zero-second visibility and did not delete the message.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/ingest-observation.json")
    try:
        run(output)
        return 0
    except (ClientError, BotoCoreError, RuntimeError, ValueError, OSError) as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
