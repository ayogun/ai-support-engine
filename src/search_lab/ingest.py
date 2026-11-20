from urllib.parse import unquote_plus

from search_lab.core import MAX_DOCUMENT_BYTES, client, embed_text, settings, vector_record


def handler(event, context):
    config = settings()
    storage = client("s3")
    vectors = client("s3vectors")
    processed = 0
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        item = record["s3"]["object"]
        key = unquote_plus(item["key"])
        if bucket != config["document_bucket"] or not key.startswith("documents/") or not key.endswith(".txt"):
            raise ValueError("Unexpected source bucket or object key.")
        version = item.get("versionId")
        request = {"Bucket": bucket, "Key": key}
        if version:
            request["VersionId"] = version
        response = storage.get_object(**request)
        raw = response["Body"].read(MAX_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise ValueError("Document exceeds the lab size limit.")
        text = raw.decode("utf-8")
        embedding = embed_text(text, endpoint_name=config["endpoint_name"], input_type="document")
        vectors.put_vectors(
            vectorBucketName=config["vector_bucket"],
            indexName=config["index_name"],
            vectors=[vector_record(key, text, embedding, str(version or ""))],
        )
        processed += 1
    return {"processed": processed}
