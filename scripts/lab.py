#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError, ClientError, WaiterError


ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".lab-state.json"
PROJECT = "hybrid-search-lab"
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_FILES = {
    "config.json": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    "model.safetensors": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
IMAGE_DIGEST = "sha256:603042f9c6d319824500988b0d7309cc094b75201da96911df7c3ddccd444ec5"
IMAGE_REPOSITORY = "huggingface-pytorch-inference"
EXPECTED_TYPES = {
    "AWS::SQS::Queue", "AWS::S3Vectors::VectorBucket", "AWS::Logs::LogGroup",
    "AWS::IAM::Role", "AWS::Lambda::Function", "AWS::Lambda::Permission",
    "AWS::S3::Bucket", "AWS::S3::BucketPolicy", "AWS::ApiGatewayV2::Api",
    "AWS::ApiGatewayV2::Integration", "AWS::ApiGatewayV2::Route", "AWS::ApiGatewayV2::Stage",
    "AWS::SageMaker::Model", "AWS::SageMaker::EndpointConfig", "AWS::SageMaker::Endpoint",
}


def write_state(state, create=False):
    payload = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()
    if create:
        descriptor = os.open(STATE_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    else:
        STATE_FILE.write_bytes(payload)
        STATE_FILE.chmod(0o600)


def load_state(session):
    if not STATE_FILE.exists():
        raise RuntimeError("No lab state exists. Deploy first or inspect your working directory.")
    state = json.loads(STATE_FILE.read_text())
    account = session.client("sts").get_caller_identity()["Account"]
    if account != state["account"]:
        raise RuntimeError("The active AWS account does not match the recorded lab account.")
    return state


def session_for_state():
    if not STATE_FILE.exists():
        raise RuntimeError("No lab state exists.")
    region = json.loads(STATE_FILE.read_text())["region"]
    return boto3.Session(region_name=region)


def has_tags(tags, lab_id):
    return tags.get("Project") == PROJECT and tags.get("LabId") == lab_id


def package_code():
    build = ROOT / ".build"
    stage = build / "package"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-compile", "--target", str(stage), "-r", str(ROOT / "requirements.txt")],
        check=True,
    )
    shutil.copytree(ROOT / "src" / "search_lab", stage / "search_lab", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    archive = build / "lambda.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(stage.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
                output.write(path, path.relative_to(stage))
    return archive


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model_file(filename, expected_hash, destination):
    if destination.exists() and sha256_file(destination) == expected_hash:
        return
    partial = destination.with_name(destination.name + ".part")
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            if urllib.parse.urlparse(response.geturl()).scheme != "https":
                raise RuntimeError("The model download redirected to an insecure URL.")
            size = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > 160 * 1024 * 1024:
                    raise RuntimeError("A model file exceeded the download size limit.")
                output.write(block)
        if sha256_file(partial) != expected_hash:
            raise RuntimeError(f"Model file checksum mismatch: {filename}.")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def package_embedding_model():
    directory = ROOT / ".build" / "model-source"
    directory.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in MODEL_FILES.items():
        print("Checking model file", filename, flush=True)
        download_model_file(filename, expected_hash, directory / filename)
    archive = ROOT / ".build" / "bge-model.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for filename in MODEL_FILES:
            output.add(directory / filename, arcname=filename, recursive=False)
        output.add(ROOT / "src" / "model" / "inference.py", arcname="code/inference.py", recursive=False)
    return archive


def resolve_inference_image(session, region):
    model = session.client("sagemaker").describe_hub_content(
        HubName="SageMakerPublicHub", HubContentType="Model",
        HubContentName="huggingface-sentencesimilarity-bge-small-en-v1-5", HubContentVersion="1.1.21",
    )
    document = json.loads(model["HubContentDocument"])
    image = document["HostingInstanceTypeVariants"]["Variants"]["c6i"]["Properties"]["ImageUri"]
    registry = image.split(".", 1)[0]
    if len(registry) != 12 or not registry.isdigit() or f".dkr.ecr.{region}.amazonaws.com/{IMAGE_REPOSITORY}:" not in image:
        raise RuntimeError("The SageMaker catalog returned an unexpected CPU image repository.")
    result = session.client("ecr").batch_get_image(
        registryId=registry, repositoryName=IMAGE_REPOSITORY, imageIds=[{"imageDigest": IMAGE_DIGEST}],
    )
    if not result.get("images"):
        raise RuntimeError("The pinned CPU serving image is unavailable in this Region; use us-west-2.")
    return registry, f"{registry}.dkr.ecr.{region}.amazonaws.com/{IMAGE_REPOSITORY}@{IMAGE_DIGEST}"


def check_serverless_quota(session):
    service = session.client("service-quotas")
    endpoint_limit = service.get_service_quota(ServiceCode="sagemaker", QuotaCode="L-99AD19BF")["Quota"]["Value"]
    concurrency_limit = service.get_service_quota(ServiceCode="sagemaker", QuotaCode="L-96300102")["Quota"]["Value"]
    if endpoint_limit < 1 or concurrency_limit < 1:
        raise RuntimeError("SageMaker Serverless Inference has no endpoint or concurrency quota in this Region.")
    print(f"SageMaker serverless limits: {endpoint_limit:g} endpoints, {concurrency_limit:g} total concurrent requests.")
    print("Limits do not guarantee unused capacity; deployment and invocation will verify it.")


def check_sdk_compatibility(session):
    model = session.client("s3vectors").meta.service_model
    operations = set(model.operation_names)
    if "PutVectorBucketDefaultIndexMode" not in operations or "QueryVectors" not in operations:
        raise RuntimeError("The active boto3/botocore is too old for S3 Vectors hybrid search; install requirements.txt in the active environment.")
    query = model.operation_model("QueryVectors").input_shape.members
    create_index = model.operation_model("CreateIndex").input_shape.members
    metadata = create_index["metadataConfiguration"].members
    if not {"queryMode", "queryTexts", "returnScores"}.issubset(query) or "textSearchConfigurations" not in metadata:
        raise RuntimeError("The active boto3/botocore lacks enhanced S3 Vectors fields; install requirements.txt in the active environment.")


def create_code_bucket(storage, state):
    request = {"Bucket": state["code_bucket"]}
    if state["region"] != "us-east-1":
        request["CreateBucketConfiguration"] = {"LocationConstraint": state["region"]}
    storage.create_bucket(**request)
    storage.put_public_access_block(
        Bucket=state["code_bucket"],
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    storage.put_bucket_encryption(
        Bucket=state["code_bucket"],
        ServerSideEncryptionConfiguration={"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
    )
    storage.put_bucket_tagging(
        Bucket=state["code_bucket"],
        Tagging={"TagSet": [{"Key": "Project", "Value": PROJECT}, {"Key": "LabId", "Value": state["lab_id"]}]},
    )
    bucket_arn = f"arn:{state['partition']}:s3:::{state['code_bucket']}"
    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*", "Action": "s3:*",
            "Resource": [bucket_arn, bucket_arn + "/*"],
            "Condition": {"Bool": {"aws:SecureTransport": "false"}},
        }],
    }
    storage.put_bucket_policy(Bucket=state["code_bucket"], Policy=json.dumps(policy))


def create_index(vectors, state, index_name, dimensions):
    return vectors.create_index(
        vectorBucketName=state["vector_bucket"], indexName=index_name, dataType="float32",
        dimension=dimensions, distanceMetric="cosine",
        metadataConfiguration={
            "nonFilterableMetadataKeys": ["text"],
            "textSearchConfigurations": [{"metadataKey": "text"}],
        },
        tags={"Project": PROJECT, "LabId": state["lab_id"]},
    )


def inspect_validation(cloudformation, change_set):
    findings = []
    paginator = cloudformation.get_paginator("describe_events")
    for page in paginator.paginate(ChangeSetName=change_set):
        for event in page.get("OperationEvents", []):
            if event.get("EventType") != "VALIDATION_ERROR":
                continue
            findings.append(event)
            print(
                "  Validation:", event.get("ValidationName", "unknown"),
                event.get("ValidationStatus", "unknown"),
                event.get("ValidationFailureMode", "unknown"),
                event.get("LogicalResourceId", ""),
                event.get("ValidationPath", ""),
                event.get("ValidationStatusReason", ""),
            )
    if not findings:
        print("  CloudFormation reported no validation errors.")
    if any(item.get("ValidationStatus") == "FAILED" and item.get("ValidationFailureMode") == "FAIL" for item in findings):
        raise RuntimeError("CloudFormation pre-deployment validation failed; the change set was not executed.")
    return findings


def deploy(arguments):
    if STATE_FILE.exists():
        raise RuntimeError("A lab state file already exists; inspect or clean up that lab first.")
    session = boto3.Session(region_name=arguments.region)
    check_sdk_compatibility(session)
    identity = session.client("sts").get_caller_identity()
    check_serverless_quota(session)
    registry, inference_image = resolve_inference_image(session, arguments.region)
    lab_id = uuid.uuid4().hex[:12]
    account = identity["Account"]
    state = {
        "account": account, "partition": identity["Arn"].split(":")[1], "region": arguments.region,
        "lab_id": lab_id, "stack": f"hybrid-search-{lab_id}", "index": "documents",
        "document_bucket": f"hybrid-docs-{account}-{lab_id}",
        "vector_bucket": f"hybrid-vectors-{account}-{lab_id}",
        "code_bucket": f"hybrid-code-{account}-{lab_id}",
        "indexes": ["documents", "bge-evaluation"],
    }
    write_state(state, create=True)
    print("Packaging Lambda code and the pinned AWS SDK...", flush=True)
    archive = package_code()
    print("Packaging the pinned BGE embedding model...", flush=True)
    model_archive = package_embedding_model()
    storage = session.client("s3")
    create_code_bucket(storage, state)
    code_key = "releases/lambda.zip"
    storage.upload_file(str(archive), state["code_bucket"], code_key)
    storage.upload_file(str(model_archive), state["code_bucket"], "releases/bge-model.tar.gz")
    cloudformation = session.client("cloudformation")
    parameters = [
        {"ParameterKey": "CodeBucket", "ParameterValue": state["code_bucket"]},
        {"ParameterKey": "CodeKey", "ParameterValue": code_key},
        {"ParameterKey": "DocumentBucketName", "ParameterValue": state["document_bucket"]},
        {"ParameterKey": "VectorBucketName", "ParameterValue": state["vector_bucket"]},
        {"ParameterKey": "LabId", "ParameterValue": lab_id},
        {"ParameterKey": "IndexName", "ParameterValue": state["index"]},
        {"ParameterKey": "InferenceRegistry", "ParameterValue": registry},
        {"ParameterKey": "InferenceImageUri", "ParameterValue": inference_image},
    ]
    change_set = cloudformation.create_change_set(
        StackName=state["stack"], ChangeSetName="create-lab", ChangeSetType="CREATE",
        TemplateBody=(ROOT / "template.yaml").read_text(), Parameters=parameters,
        Capabilities=["CAPABILITY_IAM"],
        Tags=[{"Key": "Project", "Value": PROJECT}, {"Key": "LabId", "Value": lab_id}],
    )["Id"]
    try:
        cloudformation.get_waiter("change_set_create_complete").wait(ChangeSetName=change_set)
    except WaiterError:
        inspect_validation(cloudformation, change_set)
        raise
    inspect_validation(cloudformation, change_set)
    changes = cloudformation.describe_change_set(ChangeSetName=change_set)["Changes"]
    for entry in changes:
        change = entry["ResourceChange"]
        if change["Action"] != "Add" or change["ResourceType"] not in EXPECTED_TYPES:
            raise RuntimeError("Unexpected change set; nothing was executed.")
        print(f"  {change['Action']:5} {change['LogicalResourceId']} ({change['ResourceType']})")
    print("Creating the disposable stack...", flush=True)
    cloudformation.execute_change_set(ChangeSetName=change_set)
    cloudformation.get_waiter("stack_create_complete").wait(StackName=state["stack"], WaiterConfig={"Delay": 10, "MaxAttempts": 90})
    outputs = cloudformation.describe_stacks(StackName=state["stack"])["Stacks"][0].get("Outputs", [])
    state["outputs"] = {entry["OutputKey"]: entry["OutputValue"] for entry in outputs}
    write_state(state)
    vectors = session.client("s3vectors")
    vectors.put_vector_bucket_default_index_mode(vectorBucketName=state["vector_bucket"], defaultIndexMode="ENHANCED")
    create_index(vectors, state, state["index"], 384)
    print("Enhanced index created. Upload synthetic documents with the seed command.")
    print("Search URL:", state["outputs"]["SearchUrl"])


def vector_matches_version(vectors, state, key, version):
    response = vectors.get_vectors(
        vectorBucketName=state["vector_bucket"], indexName=state["index"],
        keys=[hashlib.sha256(key.encode("utf-8")).hexdigest()], returnMetadata=True,
    )
    return any(item.get("metadata", {}).get("source_version") == str(version or "") for item in response.get("vectors", []))


def wait_for_ingest(vectors, state, key, version, timeout=240):
    deadline = time.monotonic() + timeout
    while not vector_matches_version(vectors, state, key, version):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Ingestion did not complete for {key}; inspect the failure queue and logs.")
        time.sleep(2)


def seed(arguments):
    session = session_for_state()
    state = load_state(session)
    storage = session.client("s3")
    vectors = session.client("s3vectors")
    for path in sorted((ROOT / "fixtures" / "documents").glob("*.txt")):
        body = path.read_bytes()
        key = f"documents/{path.name}"
        try:
            created = storage.put_object(Bucket=state["document_bucket"], Key=key, Body=body, ContentType="text/plain; charset=utf-8", IfNoneMatch="*")
            version = created.get("VersionId")
            print("Uploaded", path.name)
        except ClientError as error:
            if error.response["Error"]["Code"] != "PreconditionFailed":
                raise
            existing = storage.get_object(Bucket=state["document_bucket"], Key=key)
            if existing["Body"].read() != body:
                raise RuntimeError(f"Refusing to overwrite immutable object {key}.") from error
            version = existing.get("VersionId")
            if vector_matches_version(vectors, state, key, version):
                print("Already indexed", path.name)
                continue
            item = {"key": urllib.parse.quote_plus(key)}
            if version:
                item["versionId"] = version
            event = {"Records": [{"s3": {"bucket": {"name": state["document_bucket"]}, "object": item}}]}
            result = session.client("lambda").invoke(
                FunctionName=state["outputs"]["IngestFunctionName"], InvocationType="Event", Payload=json.dumps(event).encode(),
            )
            if result.get("StatusCode") != 202:
                raise RuntimeError(f"Could not requeue ingestion for {key}.")
            print("Already present; requeued ingestion for", path.name)
        wait_for_ingest(vectors, state, key, version)
        print("Indexed", path.name, flush=True)


def request_api(session, state, question, mode, limit):
    payload = json.dumps({"document_id": question, "mode": "exact"} if mode == "exact" else
                         {"question": question, "mode": mode, "limit": limit}).encode()
    request = AWSRequest(method="POST", url=state["outputs"]["SearchUrl"], data=payload, headers={"Content-Type": "application/json"})
    SigV4Auth(session.get_credentials().get_frozen_credentials(), "execute-api", state["region"]).add_auth(request)
    prepared = request.prepare()
    http_request = urllib.request.Request(prepared.url, data=prepared.body, headers=dict(prepared.headers), method="POST")
    try:
        with urllib.request.urlopen(http_request, timeout=35) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def query(arguments):
    session = session_for_state()
    state = load_state(session)
    status, payload = request_api(session, state, arguments.question, arguments.mode, arguments.limit)
    print(json.dumps({"http_status": status, **payload}, indent=2))
    if status >= 400:
        raise RuntimeError("Search request failed.")


def status(arguments):
    session = session_for_state()
    state = load_state(session)
    stack = session.client("cloudformation").describe_stacks(StackName=state["stack"])["Stacks"][0]
    result = {"stack": state["stack"], "status": stack["StackStatus"], "outputs": state.get("outputs", {})}
    if state.get("outputs") and stack["StackStatus"] in {"CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"}:
        vectors = session.client("s3vectors")
        try:
            paginator = vectors.get_paginator("list_vectors")
            result["ingested_vectors"] = sum(
                len(page.get("vectors", [])) for page in paginator.paginate(vectorBucketName=state["vector_bucket"], indexName=state["index"])
            )
        except ClientError as error:
            if error_code(error) not in {"NotFoundException", "ResourceNotFoundException"}:
                raise
            result["ingested_vectors"] = None
        attributes = session.client("sqs").get_queue_attributes(
            QueueUrl=state["outputs"]["FailedIngestQueueUrl"], AttributeNames=["ApproximateNumberOfMessages"],
        )["Attributes"]
        result["visible_failed_ingest_messages_approx"] = int(attributes["ApproximateNumberOfMessages"])
    print(json.dumps(result, indent=2))


def error_code(error):
    return error.response.get("Error", {}).get("Code", "")


def bucket_tags(storage, bucket):
    try:
        tags = storage.get_bucket_tagging(Bucket=bucket)["TagSet"]
        return {entry["Key"]: entry["Value"] for entry in tags}
    except ClientError as error:
        if error_code(error) in {"NoSuchBucket", "404"}:
            return None
        raise


def inspect_bucket_contents(storage, bucket, allowed_prefix):
    paginator = storage.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket):
        entries = page.get("Versions", []) + page.get("DeleteMarkers", [])
        if any(not entry["Key"].startswith(allowed_prefix) for entry in entries):
            raise RuntimeError(f"Refusing to remove unexpected objects from {bucket}.")


def empty_bucket(storage, bucket, allowed_prefix):
    while True:
        page = storage.list_object_versions(Bucket=bucket, MaxKeys=1000)
        entries = page.get("Versions", []) + page.get("DeleteMarkers", [])
        if not entries:
            return
        if any(not entry["Key"].startswith(allowed_prefix) for entry in entries):
            raise RuntimeError(f"Refusing to remove unexpected objects from {bucket}.")
        result = storage.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": entry["Key"], "VersionId": entry["VersionId"]} for entry in entries]})
        if result.get("Errors"):
            raise RuntimeError(f"Could not delete all objects from {bucket}: {result['Errors']}")


def cleanup(arguments):
    session = session_for_state()
    state = load_state(session)
    storage = session.client("s3")
    cloudformation = session.client("cloudformation")
    vectors = session.client("s3vectors")
    try:
        stack = cloudformation.describe_stacks(StackName=state["stack"])["Stacks"][0]
    except ClientError as error:
        if error_code(error) != "ValidationError" or "does not exist" not in str(error):
            raise
        stack = None
    if stack:
        tags = {entry["Key"]: entry["Value"] for entry in stack.get("Tags", [])}
        if not has_tags(tags, state["lab_id"]):
            raise RuntimeError("Stack ownership tags do not match; refusing cleanup.")
    try:
        vector_bucket = vectors.get_vector_bucket(vectorBucketName=state["vector_bucket"])["vectorBucket"]
    except ClientError as error:
        if error_code(error) not in {"NotFoundException", "ResourceNotFoundException"}:
            raise
        vector_bucket = None
    indexes_to_delete = []
    if vector_bucket:
        tags = vectors.list_tags_for_resource(resourceArn=vector_bucket["vectorBucketArn"])["tags"]
        if not has_tags(tags, state["lab_id"]):
            raise RuntimeError("Vector-bucket ownership tags do not match; refusing cleanup.")
        for index_name in state["indexes"]:
            try:
                index = vectors.get_index(vectorBucketName=state["vector_bucket"], indexName=index_name)["index"]
            except ClientError as error:
                if error_code(error) in {"NotFoundException", "ResourceNotFoundException"}:
                    continue
                raise
            tags = vectors.list_tags_for_resource(resourceArn=index["indexArn"])["tags"]
            if not has_tags(tags, state["lab_id"]):
                raise RuntimeError(f"Index ownership tags do not match for {index_name}.")
            indexes_to_delete.append(index_name)
    document_tags = bucket_tags(storage, state["document_bucket"])
    if document_tags is not None:
        if not has_tags(document_tags, state["lab_id"]):
            raise RuntimeError("Document-bucket ownership tags do not match; refusing cleanup.")
        inspect_bucket_contents(storage, state["document_bucket"], "documents/")
    code_tags = bucket_tags(storage, state["code_bucket"])
    if code_tags is not None:
        if not has_tags(code_tags, state["lab_id"]):
            raise RuntimeError("Code-bucket ownership tags do not match; refusing cleanup.")
        inspect_bucket_contents(storage, state["code_bucket"], "releases/")

    for index_name in indexes_to_delete:
        vectors.delete_index(vectorBucketName=state["vector_bucket"], indexName=index_name)
        print("Deleted index", index_name)
    if document_tags is not None:
        empty_bucket(storage, state["document_bucket"], "documents/")
    if stack:
        print("Deleting the tagged lab stack...", flush=True)
        cloudformation.delete_stack(StackName=state["stack"])
        cloudformation.get_waiter("stack_delete_complete").wait(StackName=state["stack"], WaiterConfig={"Delay": 10, "MaxAttempts": 90})
    elif vector_bucket:
        vectors.delete_vector_bucket(vectorBucketName=state["vector_bucket"])
    if not stack and document_tags is not None:
        storage.delete_bucket(Bucket=state["document_bucket"])
    if code_tags is not None:
        empty_bucket(storage, state["code_bucket"], "releases/")
        storage.delete_bucket(Bucket=state["code_bucket"])
    STATE_FILE.unlink()
    print("Tagged lab resources deleted.")


def main():
    parser = argparse.ArgumentParser(description="Deploy, exercise, and clean up the disposable hybrid-search lab.")
    commands = parser.add_subparsers(dest="command", required=True)
    deployment = commands.add_parser("deploy")
    deployment.add_argument("--region", default="us-west-2")
    deployment.set_defaults(run=deploy)
    commands.add_parser("seed").set_defaults(run=seed)
    search_parser = commands.add_parser("query")
    search_parser.add_argument("question")
    search_parser.add_argument("--mode", choices=["keyword", "vector", "hybrid", "exact"], default="hybrid")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.set_defaults(run=query)
    commands.add_parser("status").set_defaults(run=status)
    commands.add_parser("cleanup").set_defaults(run=cleanup)
    arguments = parser.parse_args()
    try:
        arguments.run(arguments)
        return 0
    except (ClientError, BotoCoreError, RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        if STATE_FILE.exists():
            print("Lab state was retained; inspect it and run cleanup when appropriate.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
