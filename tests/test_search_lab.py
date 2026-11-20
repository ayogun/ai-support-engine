import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from botocore.exceptions import ClientError

from model import inference
from search_lab import api, core, ingest
from scripts import evaluate, lab


class CoreTests(unittest.TestCase):
    def test_validate_search(self):
        self.assertEqual(core.validate_search(" hello ", "hybrid", 3), "hello")
        invalid = [
            (" ", "hybrid", 1),
            (None, "hybrid", 1),
            ("é" * 501, "hybrid", 1),
            ("hello", "unknown", 1),
            ("hello", "vector", True),
            ("hello", "vector", 0),
            ("hello", "vector", 11),
        ]
        for question, mode, limit in invalid:
            with self.subTest(question=question, mode=mode, limit=limit):
                with self.assertRaises(ValueError):
                    core.validate_search(question, mode, limit)

    def test_three_query_shapes(self):
        for mode in ("keyword", "vector", "hybrid"):
            request = core.query_request("bucket", "index", " AUTH482 ", mode, 3, [0.1, 0.2])
            self.assertEqual(request["queryMode"], "ENHANCED")
            self.assertEqual(request["topK"], 3)
            self.assertEqual("queryTexts" in request, mode in ("keyword", "hybrid"))
            self.assertEqual("queryVector" in request, mode in ("vector", "hybrid"))
            if "queryTexts" in request:
                self.assertEqual(request["queryTexts"], [{"metadataKey": "text", "text": "AUTH482"}])
        with self.assertRaises(ValueError):
            core.query_request("bucket", "index", "hello", "hybrid", 3)

    def test_embedding_payload_and_shape(self):
        runtime = Mock()
        runtime.invoke_endpoint.return_value = {"Body": io.BytesIO(b'{"embedding":[0.1,0.2]}')}
        self.assertEqual(core.embed_text("hello", "endpoint", input_type="query", dimensions=2, runtime=runtime), [0.1, 0.2])
        self.assertEqual(json.loads(runtime.invoke_endpoint.call_args.kwargs["Body"]), {"text": "hello", "input_type": "query"})
        self.assertEqual(runtime.invoke_endpoint.call_args.kwargs["EndpointName"], "endpoint")
        for vector in ([0.1], [float("nan"), 0.2], [True, 0.2]):
            runtime.invoke_endpoint.return_value = {"Body": io.BytesIO(json.dumps({"embedding": vector}).encode())}
            with self.subTest(vector=vector), self.assertRaises(RuntimeError):
                core.embed_text("hello", "endpoint", dimensions=2, runtime=runtime)
        runtime.invoke_endpoint.return_value = {"Body": io.BytesIO(b'{}')}
        with self.assertRaises(RuntimeError):
            core.embed_text("hello", "endpoint", dimensions=2, runtime=runtime)
        with self.assertRaises(ValueError):
            core.embed_text("  ", "endpoint", runtime=runtime)
        with self.assertRaises(ValueError):
            core.embed_text("hello", "endpoint", input_type="unknown", runtime=runtime)

    def test_vector_record_and_limits(self):
        record = core.vector_record("documents/auth482.txt", "Example", [0.1, 0.2], "v1")
        self.assertEqual(record["metadata"]["document_id"], "auth482")
        self.assertEqual(record["metadata"]["source_version"], "v1")
        self.assertEqual(len(record["key"]), 64)
        self.assertEqual(record["key"], core.vector_key("documents/auth482.txt"))
        for text in (" ", "é" * (core.MAX_DOCUMENT_BYTES // 2 + 1)):
            with self.subTest(text=text[:4]), self.assertRaises(ValueError):
                core.vector_record("documents/item.txt", text, [0.1])

    def test_search_does_not_embed_keyword_queries(self):
        vectors = Mock()
        vectors.query_vectors.return_value = {"vectors": [{"metadata": {"document_id": "auth482", "text": "Example"}}]}
        runtime = Mock()
        result = core.search("AUTH482", "keyword", 1, {"vector_bucket": "bucket", "index_name": "index", "endpoint_name": "endpoint"}, vectors, runtime)
        self.assertEqual(result["results"][0]["document_id"], "auth482")
        runtime.invoke_endpoint.assert_not_called()
        self.assertNotIn("queryVector", vectors.query_vectors.call_args.kwargs)

    @patch("search_lab.core.embed_text", return_value=[0.1, 0.2])
    def test_search_embeds_queries_as_queries(self, embed):
        vectors = Mock()
        vectors.query_vectors.return_value = {"vectors": []}
        runtime = Mock()
        core.search("hello", "hybrid", 1, {"vector_bucket": "bucket", "index_name": "index", "endpoint_name": "endpoint"}, vectors, runtime)
        embed.assert_called_once_with("hello", endpoint_name="endpoint", input_type="query", runtime=runtime)

    def test_exact_lookup_uses_case_sensitive_source_key_without_embedding(self):
        vectors = Mock()
        source_key = "documents/Auth-4.01_v2.txt"
        key = core.vector_key(source_key)
        vectors.get_vectors.return_value = {"vectors": [{"key": key, "metadata": {
            "document_id": "Auth-4.01_v2", "source_key": source_key, "text": "Example",
        }}]}
        config = {"vector_bucket": "bucket", "index_name": "index"}
        result = core.exact_lookup("Auth-4.01_v2", config, vectors)
        self.assertEqual(result["results"][0]["document_id"], "Auth-4.01_v2")
        vectors.get_vectors.assert_called_once_with(
            vectorBucketName="bucket", indexName="index", keys=[key], returnMetadata=True,
        )
        self.assertNotEqual(core.vector_key(source_key), core.vector_key("documents/auth-4.01_v2.txt"))
        vectors.query_vectors.assert_not_called()

    def test_exact_lookup_missing_invalid_and_inconsistent_metadata(self):
        vectors = Mock()
        vectors.get_vectors.return_value = {"vectors": []}
        config = {"vector_bucket": "bucket", "index_name": "index"}
        self.assertEqual(core.exact_lookup("missing-01", config, vectors)["results"], [])
        for document_id in (None, "", "../secret", ".hidden", "two words", "é", "a" * 129):
            with self.subTest(document_id=document_id), self.assertRaises(ValueError):
                core.exact_lookup(document_id, config, vectors)
        vectors.get_vectors.return_value = {"vectors": [{"key": core.vector_key("documents/a.txt"),
            "metadata": {"document_id": "b", "source_key": "documents/a.txt"}}]}
        with self.assertRaises(RuntimeError):
            core.exact_lookup("a", config, vectors)

    @patch("search_lab.core.boto3.client")
    @patch.dict("os.environ", {"MODEL_MAX_ATTEMPTS": "3", "MODEL_READ_TIMEOUT": "22"})
    def test_model_retry_attempts_are_bounded(self, boto_client):
        core.client.cache_clear()
        try:
            core.client("sagemaker-runtime")
            config = boto_client.call_args.kwargs["config"]
            self.assertEqual(config.retries["total_max_attempts"], 3)
            self.assertEqual(config.read_timeout, 22)
        finally:
            core.client.cache_clear()
        with patch.dict("os.environ", {"MODEL_MAX_ATTEMPTS": "4"}), self.assertRaises(RuntimeError):
            core.client("sagemaker-runtime")
        core.client.cache_clear()


class ModelHandlerTests(unittest.TestCase):
    def test_query_instruction_and_document_text(self):
        self.assertEqual(inference.prepare_text({"text": "hello", "input_type": "document"}), "hello")
        self.assertEqual(inference.prepare_text({"text": "hello", "input_type": "query"}), inference.QUERY_PREFIX + "hello")
        for payload in (None, {}, {"text": " ", "input_type": "query"}, {"text": "é" * 4001, "input_type": "document"}, {"text": "x", "input_type": "other"}):
            with self.subTest(payload=repr(payload)[:30]), self.assertRaises(ValueError):
                inference.prepare_text(payload)

    def test_handler_tokenizes_query_and_returns_normalized_vector(self):
        torch = MagicMock()
        model = Mock()
        tokenizer = Mock(return_value={"input_ids": "tokens"})
        model.return_value = SimpleNamespace(last_hidden_state=MagicMock())
        torch.nn.functional.normalize.return_value.__getitem__.return_value.tolist.return_value = [0.6, 0.8]
        result = inference.transform_fn(
            {"torch": torch, "model": model, "tokenizer": tokenizer},
            json.dumps({"text": "hello", "input_type": "query"}), "application/json", "application/json",
        )
        self.assertEqual(json.loads(result), {"embedding": [0.6, 0.8]})
        tokenizer.assert_called_once_with(inference.QUERY_PREFIX + "hello", padding=True, truncation=True, max_length=512, return_tensors="pt")
        model.assert_called_once_with(input_ids="tokens")
        with self.assertRaises(ValueError):
            inference.transform_fn({}, "{}", "text/plain", "application/json")


class ApiTests(unittest.TestCase):
    @patch("search_lab.api.settings", return_value={})
    @patch("search_lab.api.search", return_value={"mode": "hybrid", "results": []})
    def test_base64_request(self, search, settings):
        body = base64.b64encode(b'{"question":"hello"}').decode()
        response = api.handler({"body": body, "isBase64Encoded": True}, None)
        self.assertEqual(response["statusCode"], 200)
        search.assert_called_once_with("hello", "hybrid", 5, {})

    def test_bad_requests(self):
        for event in ({"body": "{"}, {"body": "[]"}, {"body": "!!!", "isBase64Encoded": True}, {"body": "x" * 4097}):
            with self.subTest(event=event):
                self.assertEqual(api.handler(event, None)["statusCode"], 400)

    @patch("search_lab.api.settings", return_value={})
    @patch("search_lab.api.exact_lookup", return_value={"mode": "exact", "results": []})
    def test_exact_request_uses_document_id_not_question(self, lookup, settings):
        response = api.handler({"body": '{"mode":"exact","document_id":"Case-1.0"}'}, None)
        self.assertEqual(response["statusCode"], 200)
        lookup.assert_called_once_with("Case-1.0", {})

    @patch("search_lab.api.settings", return_value={})
    @patch("search_lab.api.search")
    def test_dependency_errors_are_not_reported_as_user_errors(self, search, settings):
        for failure in (RuntimeError("bad embedding"), ClientError({"Error": {"Code": "AccessDeniedException", "Message": "private detail"}}, "QueryVectors")):
            search.side_effect = failure
            with self.assertLogs("search_lab.api", level="WARNING") as logs:
                response = api.handler({"body": '{"question":"hello"}'}, None)
            self.assertEqual(response["statusCode"], 502)
            self.assertNotIn("private detail", response["body"])
            self.assertNotIn("private detail", " ".join(logs.output))
            if isinstance(failure, ClientError):
                self.assertIn("operation=QueryVectors", " ".join(logs.output))


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.storage = Mock()
        self.vectors = Mock()
        self.storage.get_object.return_value = {"Body": io.BytesIO(b"Example text")}
        self.config = {"document_bucket": "docs", "vector_bucket": "vectors", "index_name": "index", "endpoint_name": "endpoint"}
        self.event = {"Records": [{"s3": {"bucket": {"name": "docs"}, "object": {"key": "documents%2Fhello+world.txt", "versionId": "v1"}}}]}

    @patch("search_lab.ingest.embed_text", return_value=[0.1, 0.2])
    def test_event_decodes_key_and_fetches_version(self, embed):
        with patch("search_lab.ingest.settings", return_value=self.config), patch("search_lab.ingest.client", side_effect=[self.storage, self.vectors]):
            self.assertEqual(ingest.handler(self.event, None), {"processed": 1})
        self.storage.get_object.assert_called_once_with(Bucket="docs", Key="documents/hello world.txt", VersionId="v1")
        embed.assert_called_once_with("Example text", endpoint_name="endpoint", input_type="document")
        vector = self.vectors.put_vectors.call_args.kwargs["vectors"][0]
        self.assertEqual(vector["metadata"]["source_version"], "v1")
        self.assertEqual(vector["metadata"]["text"], "Example text")

    @patch("search_lab.ingest.embed_text")
    def test_rejects_wrong_bucket_and_oversized_document(self, embed):
        self.event["Records"][0]["s3"]["bucket"]["name"] = "other"
        with patch("search_lab.ingest.settings", return_value=self.config), patch("search_lab.ingest.client", side_effect=[self.storage, self.vectors]):
            with self.assertRaises(ValueError):
                ingest.handler(self.event, None)
        self.event["Records"][0]["s3"]["bucket"]["name"] = "docs"
        self.storage.get_object.return_value = {"Body": io.BytesIO(b"x" * (core.MAX_DOCUMENT_BYTES + 1))}
        with patch("search_lab.ingest.settings", return_value=self.config), patch("search_lab.ingest.client", side_effect=[self.storage, self.vectors]):
            with self.assertRaises(ValueError):
                ingest.handler(self.event, None)
        embed.assert_not_called()
        self.vectors.put_vectors.assert_not_called()


class LabSafetyTests(unittest.TestCase):
    def test_seed_requeues_unchanged_document_after_prior_failure(self):
        session = Mock()
        storage, functions, vectors = Mock(), Mock(), Mock()
        session.client.side_effect = lambda name: {"s3": storage, "lambda": functions, "s3vectors": vectors}[name]
        storage.put_object.side_effect = ClientError({"Error": {"Code": "PreconditionFailed", "Message": "exists"}}, "PutObject")
        storage.get_object.return_value = {"Body": io.BytesIO(b"Example"), "VersionId": "v1"}
        functions.invoke.return_value = {"StatusCode": 202}
        state = {"document_bucket": "docs", "outputs": {"IngestFunctionName": "ingest"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixtures" / "documents").mkdir(parents=True)
            (root / "fixtures" / "documents" / "hello.txt").write_text("Example")
            with patch("scripts.lab.ROOT", root), patch("scripts.lab.session_for_state", return_value=session), patch("scripts.lab.load_state", return_value=state), patch("scripts.lab.vector_matches_version", return_value=False), patch("scripts.lab.wait_for_ingest"):
                lab.seed(None)
        event = json.loads(functions.invoke.call_args.kwargs["Payload"])
        self.assertEqual(event["Records"][0]["s3"]["object"]["versionId"], "v1")
        self.assertEqual(functions.invoke.call_args.kwargs["InvocationType"], "Event")

    def test_seed_refuses_to_replace_changed_object(self):
        session = Mock()
        storage, functions, vectors = Mock(), Mock(), Mock()
        session.client.side_effect = lambda name: {"s3": storage, "lambda": functions, "s3vectors": vectors}[name]
        storage.put_object.side_effect = ClientError({"Error": {"Code": "PreconditionFailed", "Message": "exists"}}, "PutObject")
        storage.get_object.return_value = {"Body": io.BytesIO(b"Different data")}
        state = {"document_bucket": "docs", "outputs": {"IngestFunctionName": "ingest"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixtures" / "documents").mkdir(parents=True)
            (root / "fixtures" / "documents" / "hello.txt").write_text("Example")
            with patch("scripts.lab.ROOT", root), patch("scripts.lab.session_for_state", return_value=session), patch("scripts.lab.load_state", return_value=state):
                with self.assertRaises(RuntimeError):
                    lab.seed(None)
        functions.invoke.assert_not_called()

    def test_seed_skips_an_already_indexed_version(self):
        session = Mock()
        storage, functions, vectors = Mock(), Mock(), Mock()
        session.client.side_effect = lambda name: {"s3": storage, "lambda": functions, "s3vectors": vectors}[name]
        storage.put_object.side_effect = ClientError({"Error": {"Code": "PreconditionFailed", "Message": "exists"}}, "PutObject")
        storage.get_object.return_value = {"Body": io.BytesIO(b"Example"), "VersionId": "v1"}
        state = {"document_bucket": "docs", "outputs": {"IngestFunctionName": "ingest"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixtures" / "documents").mkdir(parents=True)
            (root / "fixtures" / "documents" / "hello.txt").write_text("Example")
            with patch("scripts.lab.ROOT", root), patch("scripts.lab.session_for_state", return_value=session), patch("scripts.lab.load_state", return_value=state), patch("scripts.lab.vector_matches_version", return_value=True), patch("scripts.lab.wait_for_ingest") as wait:
                lab.seed(None)
        functions.invoke.assert_not_called()
        wait.assert_not_called()

    def test_vector_version_must_match_source(self):
        vectors = Mock()
        vectors.get_vectors.return_value = {"vectors": [{"metadata": {"source_version": "v1"}}]}
        state = {"vector_bucket": "bucket", "index": "documents"}
        self.assertTrue(lab.vector_matches_version(vectors, state, "documents/item.txt", "v1"))
        self.assertFalse(lab.vector_matches_version(vectors, state, "documents/item.txt", "v2"))
        self.assertEqual(vectors.get_vectors.call_args.kwargs["keys"], [core.vector_key("documents/item.txt")])

    def test_ingestion_wait_times_out(self):
        with patch("scripts.lab.vector_matches_version", return_value=False), patch("scripts.lab.time.monotonic", side_effect=[0, 2]):
            with self.assertRaises(RuntimeError):
                lab.wait_for_ingest(Mock(), {}, "documents/item.txt", "v1", timeout=1)

    def test_evaluation_metrics_count_misses_correctly(self):
        observations = [
            {"mode": "keyword", "kind": "exact_identifier", "rank": 1, "query_ms": 10.0},
            {"mode": "keyword", "kind": "exact_identifier", "rank": 3, "query_ms": 20.0},
            {"mode": "keyword", "kind": "paraphrase", "rank": None, "query_ms": 30.0},
        ]
        result = evaluate.summarize(observations)
        self.assertEqual(result["all"]["keyword"]["queries"], 3)
        self.assertEqual(result["all"]["keyword"]["hit_at_1"], 1)
        self.assertEqual(result["all"]["keyword"]["hit_at_3"], 2)
        self.assertEqual(result["all"]["keyword"]["mrr_at_3"], 0.4444)
        self.assertEqual(result["all"]["keyword"]["query_ms_median"], 20.0)
        self.assertEqual(result["all"]["keyword"]["query_ms_p95"], 30.0)

    def test_predeployment_validation_blocks_failed_findings(self):
        cloudformation = Mock()
        paginator = cloudformation.get_paginator.return_value
        paginator.paginate.return_value = [{"OperationEvents": [{"EventType": "VALIDATION_ERROR", "ValidationName": "PROPERTY_VALIDATION", "ValidationStatus": "FAILED", "ValidationFailureMode": "FAIL"}]}]
        with self.assertRaises(RuntimeError):
            lab.inspect_validation(cloudformation, "change-set")
        paginator.paginate.assert_called_once_with(ChangeSetName="change-set")

    def test_bucket_cleanup_refuses_unexpected_objects(self):
        storage = Mock()
        storage.get_paginator.return_value.paginate.return_value = [{"Versions": [{"Key": "other/valuable.txt", "VersionId": "v1"}]}]
        with self.assertRaises(RuntimeError):
            lab.inspect_bucket_contents(storage, "bucket", "documents/")
        storage.delete_objects.assert_not_called()

    def test_delete_errors_do_not_loop_forever(self):
        storage = Mock()
        storage.list_object_versions.return_value = {"Versions": [{"Key": "documents/a.txt", "VersionId": "v1"}]}
        storage.delete_objects.return_value = {"Errors": [{"Code": "AccessDenied", "Key": "documents/a.txt"}]}
        with self.assertRaises(RuntimeError):
            lab.empty_bucket(storage, "bucket", "documents/")
        storage.delete_objects.assert_called_once()

    def test_cleanup_refuses_unowned_stack_before_any_deletion(self):
        session = Mock()
        storage, cloudformation, vectors = Mock(), Mock(), Mock()
        session.client.side_effect = lambda name: {"s3": storage, "cloudformation": cloudformation, "s3vectors": vectors}[name]
        cloudformation.describe_stacks.return_value = {"Stacks": [{"Tags": [{"Key": "Project", "Value": "not-this-lab"}]}]}
        state = {"lab_id": "0123456789ab", "stack": "stack"}
        with patch("scripts.lab.session_for_state", return_value=session), patch("scripts.lab.load_state", return_value=state):
            with self.assertRaises(RuntimeError):
                lab.cleanup(None)
        cloudformation.delete_stack.assert_not_called()
        vectors.delete_index.assert_not_called()
        storage.delete_objects.assert_not_called()

    def test_cleanup_checks_all_buckets_before_deleting_indexes(self):
        session = Mock()
        storage, cloudformation, vectors = Mock(), Mock(), Mock()
        session.client.side_effect = lambda name: {"s3": storage, "cloudformation": cloudformation, "s3vectors": vectors}[name]
        tags = {"Project": lab.PROJECT, "LabId": "0123456789ab"}
        cloudformation.describe_stacks.return_value = {"Stacks": [{"Tags": [{"Key": key, "Value": value} for key, value in tags.items()]}]}
        vectors.get_vector_bucket.return_value = {"vectorBucket": {"vectorBucketArn": "bucket-arn"}}
        vectors.get_index.return_value = {"index": {"indexArn": "index-arn"}}
        vectors.list_tags_for_resource.return_value = {"tags": tags}
        state = {"lab_id": tags["LabId"], "stack": "stack", "vector_bucket": "vectors", "document_bucket": "docs", "code_bucket": "code", "indexes": ["documents"]}
        with patch("scripts.lab.session_for_state", return_value=session), patch("scripts.lab.load_state", return_value=state), patch("scripts.lab.bucket_tags", return_value={"Project": "someone-else"}):
            with self.assertRaises(RuntimeError):
                lab.cleanup(None)
        vectors.delete_index.assert_not_called()
        cloudformation.delete_stack.assert_not_called()


if __name__ == "__main__":
    unittest.main()
