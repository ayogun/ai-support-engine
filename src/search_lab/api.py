import base64
import binascii
import json
import logging

from botocore.exceptions import BotoCoreError, ClientError

from search_lab.core import exact_lookup, search, settings


logger = logging.getLogger(__name__)


def response(status, payload):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(payload),
    }


def handler(event, context):
    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body, validate=True).decode("utf-8")
        if len(body.encode("utf-8")) > 4096:
            raise ValueError("request body is too large")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        if payload.get("mode") == "exact":
            result = exact_lookup(payload.get("document_id"), settings())
        else:
            result = search(payload.get("question"), payload.get("mode", "hybrid"), payload.get("limit", 5), settings())
        return response(200, result)
    except (ValueError, UnicodeError, binascii.Error) as error:
        return response(400, {"error": str(error)})
    except (ClientError, BotoCoreError, RuntimeError) as error:
        code = error.response.get("Error", {}).get("Code", "aws_error") if isinstance(error, ClientError) else "dependency_error"
        operation = error.operation_name if isinstance(error, ClientError) else "unknown"
        logger.warning("Search dependency failed: %s (operation=%s)", code, operation)
        return response(502, {"error": "search dependency unavailable", "code": code})
