#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError, WaiterError

from scripts import lab, probe_latency


def verify_lab(session, state):
    cloudformation = session.client("cloudformation")
    stack = cloudformation.describe_stacks(StackName=state["stack"])["Stacks"][0]
    tags = {entry["Key"]: entry["Value"] for entry in stack.get("Tags", [])}
    if not lab.has_tags(tags, state["lab_id"]) or stack["StackStatus"] not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}:
        raise RuntimeError("The recorded stack is not a ready, owned lab; refusing to change its function.")
    resource = cloudformation.describe_stack_resource(StackName=state["stack"], LogicalResourceId="SearchFunction")["StackResourceDetail"]
    function_name = state["outputs"]["SearchFunctionName"]
    if resource["PhysicalResourceId"] != function_name or function_name != f"{state['stack']}-search":
        raise RuntimeError("The search function does not match the recorded lab stack.")
    sage = session.client("sagemaker")
    endpoint = sage.describe_endpoint(EndpointName=state["outputs"]["EmbeddingEndpointName"])
    variants = sage.describe_endpoint_config(EndpointConfigName=endpoint["EndpointConfigName"])["ProductionVariants"]
    if len(variants) != 1 or variants[0].get("ServerlessConfig", {}).get("MaxConcurrency") != 1:
        raise RuntimeError("This comparison requires the lab's one-concurrency serverless endpoint.")
    functions = session.client("lambda")
    configuration = functions.get_function_configuration(FunctionName=function_name)
    variables = configuration.get("Environment", {}).get("Variables", {})
    expected = {"VECTOR_BUCKET": state["vector_bucket"], "INDEX_NAME": state["index"],
                "EMBEDDING_ENDPOINT": state["outputs"]["EmbeddingEndpointName"]}
    if any(variables.get(key) != value for key, value in expected.items()):
        raise RuntimeError("The function environment does not match the recorded lab.")
    return functions, function_name, dict(variables)


def set_attempts(functions, function_name, original_variables, attempts):
    configuration = functions.get_function_configuration(FunctionName=function_name)
    current = configuration.get("Environment", {}).get("Variables", {})
    current_other = {key: value for key, value in current.items() if key != "MODEL_MAX_ATTEMPTS"}
    original_other = {key: value for key, value in original_variables.items() if key != "MODEL_MAX_ATTEMPTS"}
    if current_other != original_other:
        raise RuntimeError("The function environment changed concurrently; refusing to overwrite it.")
    desired = dict(original_variables)
    if attempts is not None:
        desired["MODEL_MAX_ATTEMPTS"] = str(attempts)
    if current == desired:
        return
    functions.update_function_configuration(
        FunctionName=function_name, Environment={"Variables": desired}, RevisionId=configuration["RevisionId"],
    )
    functions.get_waiter("function_updated_v2").wait(
        FunctionName=function_name, WaiterConfig={"Delay": 2, "MaxAttempts": 90},
    )


def log_counts(logs, state, block):
    paginator = logs.get_paginator("filter_log_events")
    counts = {"successful_invocations_with_retries": 0, "successful_invocation_retry_attempts": 0,
              "invoke_endpoint_throttles_logged": 0}
    for page in paginator.paginate(logGroupName=f"/aws/lambda/{state['stack']}-search",
                                   startTime=block["start_ms"], endTime=block["end_ms"]):
        for event in page.get("events", []):
            message = event.get("message", "")
            match = re.search(r"Embedding endpoint succeeded after (\d+) SDK retries", message)
            if match:
                counts["successful_invocations_with_retries"] += 1
                counts["successful_invocation_retry_attempts"] += int(match.group(1))
            if "Search dependency failed: ThrottlingException (operation=InvokeEndpoint)" in message:
                counts["invoke_endpoint_throttles_logged"] += 1
    return counts


def run(arguments):
    session = lab.session_for_state()
    state = lab.load_state(session)
    functions, function_name, original_variables = verify_lab(session, state)
    blocks = []
    try:
        for attempts in (1, 3, 3, 1):
            set_attempts(functions, function_name, original_variables, attempts)
            warmup = probe_latency.timed_request(session, state, "hybrid")
            warmup.pop("_started", None)
            if warmup.get("outcome") != "success" or warmup.get("first_relevant_rank") is None:
                raise RuntimeError(f"Warm-up failed or missed the expected document with {attempts} total model attempts.")
            block = {"model_total_max_attempts": attempts, "warmup": warmup,
                     "start_ms": int(time.time() * 1000), "bursts": []}
            for number in range(arguments.rounds_per_block):
                if number:
                    time.sleep(2)
                block["bursts"].append(probe_latency.burst(state, arguments.concurrency))
                print(f"Completed attempts={attempts} burst {number + 1}/{arguments.rounds_per_block}.", flush=True)
            block["end_ms"] = int(time.time() * 1000)
            blocks.append(block)
    finally:
        set_attempts(functions, function_name, original_variables, None)
    if arguments.include_logs:
        time.sleep(10)
        logs = session.client("logs")
        for block in blocks:
            try:
                block["log_counts"] = log_counts(logs, state, block)
            except (ClientError, BotoCoreError) as error:
                block["log_counts"] = {"unavailable": type(error).__name__}
    summaries = {}
    for attempts in (1, 3):
        samples = [sample for block in blocks if block["model_total_max_attempts"] == attempts
                   for burst in block["bursts"] for sample in burst["samples"]]
        summaries[str(attempts)] = probe_latency.summarize(samples)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "region": state["region"],
        "design": "ABBA blocks: SDK total attempts 1, 3, 3, 1; one warm-up per block and identical signed hybrid bursts.",
        "question": probe_latency.QUESTION, "expected_document_id": probe_latency.EXPECTED_DOCUMENT_ID,
        "top_k": 3, "endpoint_max_concurrency": 1, "concurrency": arguments.concurrency,
        "rounds_per_block": arguments.rounds_per_block, "pause_between_bursts_seconds": 2,
        "read_timeout_seconds": original_variables.get("MODEL_READ_TIMEOUT"),
        "original_model_max_attempts": original_variables.get("MODEL_MAX_ATTEMPTS"),
        "limitations": "Small, repeated-query, single-client experiment; ABBA reduces but does not eliminate time effects. HTTP success and retrieval hit are separate. Logs, when requested, are aggregated by time window and may be delayed; they are not request-level correlation. Retries can increase latency, duplicate work, and still fail or exceed the function deadline.",
        "summary": summaries, "blocks": blocks,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))
    print("Restored the original search-function environment; wrote", output)


def main():
    parser = argparse.ArgumentParser(description="Compare bounded SDK retries using an owned disposable lab, then restore its environment.")
    parser.add_argument("--concurrency", type=int, default=3, choices=range(1, 4), metavar="1-3")
    parser.add_argument("--rounds-per-block", type=int, default=3, choices=range(1, 6), metavar="1-5")
    parser.add_argument("--include-logs", action="store_true")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
        return 0
    except (ClientError, BotoCoreError, WaiterError, RuntimeError, ValueError, OSError) as error:
        print(f"Retry comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
