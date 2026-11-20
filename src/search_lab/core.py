import hashlib
import json
import logging
import math
import os
import re
from functools import lru_cache

import boto3
from botocore.config import Config


MODEL_ID = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
MAX_DOCUMENT_BYTES = 8000
MAX_QUERY_BYTES = 1000
MODES = {"keyword", "vector", "hybrid"}
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def client(service_name):
    if service_name == "sagemaker-runtime":
        try:
            attempts = int(os.environ.get("MODEL_MAX_ATTEMPTS", "1"))
        except ValueError as error:
            raise RuntimeError("Invalid model retry configuration.") from error
        if not 1 <= attempts <= 3:
            raise RuntimeError("Model retry attempts must be between 1 and 3.")
        config = Config(
            connect_timeout=3,
            read_timeout=int(os.environ.get("MODEL_READ_TIMEOUT", "90")),
            retries={"mode": "standard", "total_max_attempts": attempts},
        )
    else:
        config = Config(retries={"mode": "standard", "total_max_attempts": 4})
    return boto3.client(service_name, config=config)


def settings():
    return {
        "vector_bucket": os.environ["VECTOR_BUCKET"],
        "index_name": os.environ["INDEX_NAME"],
        "document_bucket": os.environ["DOCUMENT_BUCKET"],
        "endpoint_name": os.environ["EMBEDDING_ENDPOINT"],
    }


def embed_text(text, endpoint_name, input_type="document", dimensions=DIMENSIONS, runtime=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")
    if input_type not in {"document", "query"}:
        raise ValueError("input_type must be document or query.")
    runtime = runtime or client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps({"text": text, "input_type": input_type}),
    )
    retries = response.get("ResponseMetadata", {}).get("RetryAttempts", 0)
    if retries:
        logger.warning("Embedding endpoint succeeded after %s SDK retries", retries)
    try:
        payload = json.loads(response["Body"].read())
        vector = payload["embedding"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("The embedding model returned an invalid response.") from error
    if not isinstance(vector, list) or len(vector) != dimensions or not all(type(value) in (int, float) and math.isfinite(value) for value in vector):
        raise RuntimeError("The embedding model returned an invalid vector.")
    return vector


def vector_key(source_key):
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()


def identifier_source_key(document_id):
    if not isinstance(document_id, str) or not IDENTIFIER_PATTERN.fullmatch(document_id):
        raise ValueError("document_id must contain 1–128 ASCII letters, digits, dots, underscores, or hyphens and begin with a letter or digit")
    return f"documents/{document_id}.txt"


def vector_record(source_key, text, embedding, source_version=""):
    encoded = text.encode("utf-8")
    if not text.strip() or len(encoded) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Document must contain 1–{MAX_DOCUMENT_BYTES} UTF-8 bytes.")
    document_id = source_key.rsplit("/", 1)[-1].removesuffix(".txt")
    return {
        "key": vector_key(source_key),
        "data": {"float32": [float(value) for value in embedding]},
        "metadata": {
            "document_id": document_id,
            "source_key": source_key,
            "source_version": source_version,
            "text": text,
        },
    }


def validate_search(question, mode, limit):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if len(question.encode("utf-8")) > MAX_QUERY_BYTES:
        raise ValueError(f"question must be at most {MAX_QUERY_BYTES} UTF-8 bytes")
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("mode must be keyword, vector, or hybrid")
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("limit must be an integer from 1 to 10")
    return question.strip()


def query_request(vector_bucket, index_name, question, mode, limit, embedding=None):
    question = validate_search(question, mode, limit)
    request = {
        "vectorBucketName": vector_bucket,
        "indexName": index_name,
        "topK": limit,
        "queryMode": "ENHANCED",
        "returnMetadata": True,
        "returnDistance": True,
        "returnScores": True,
    }
    if mode in {"keyword", "hybrid"}:
        request["queryTexts"] = [{"metadataKey": "text", "text": question}]
    if mode in {"vector", "hybrid"}:
        if embedding is None:
            raise ValueError("An embedding is required for vector or hybrid search.")
        request["queryVector"] = {"float32": [float(value) for value in embedding]}
    return request


def search(question, mode, limit, config, vectors_client=None, runtime=None):
    question = validate_search(question, mode, limit)
    embedding = None
    if mode != "keyword":
        embedding = embed_text(question, endpoint_name=config["endpoint_name"], input_type="query", runtime=runtime)
    request = query_request(config["vector_bucket"], config["index_name"], question, mode, limit, embedding)
    response = (vectors_client or client("s3vectors")).query_vectors(**request)
    results = []
    for vector in response.get("vectors", []):
        metadata = vector.get("metadata", {})
        results.append({
            "document_id": metadata.get("document_id"),
            "source_key": metadata.get("source_key"),
            "text": metadata.get("text"),
            "distance": vector.get("distance"),
            "scores": vector.get("scores"),
        })
    return {"mode": mode, "results": results}


def exact_lookup(document_id, config, vectors_client=None):
    source_key = identifier_source_key(document_id)
    key = vector_key(source_key)
    response = (vectors_client or client("s3vectors")).get_vectors(
        vectorBucketName=config["vector_bucket"], indexName=config["index_name"],
        keys=[key], returnMetadata=True,
    )
    vectors = response.get("vectors", [])
    if not vectors:
        return {"mode": "exact", "results": []}
    if len(vectors) != 1 or vectors[0].get("key") != key:
        raise RuntimeError("Exact lookup returned an unexpected vector.")
    metadata = vectors[0].get("metadata", {})
    if metadata.get("document_id") != document_id or metadata.get("source_key") != source_key:
        raise RuntimeError("Exact lookup returned inconsistent document metadata.")
    return {"mode": "exact", "results": [{
        "document_id": document_id, "source_key": source_key, "text": metadata.get("text"),
    }]}
