import hashlib
import json
import tempfile
import unittest
import zipfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from scripts import compare_results, compare_retries, create_release, evaluate, evaluate_bitext, evaluate_scifact, evaluate_support, lab, probe_latency, summarize_retries, verify_release
from search_lab import core


class SciFactSelectionTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"doc_id": number, "title": f"Title {number}", "abstract": [f"Abstract {number}"]}
            for number in range(1, 8)
        ]
        self.corpus.append({"doc_id": 8, "title": "Long", "abstract": ["x" * 8001]})
        self.claims = [
            {"id": 10, "claim": "First claim", "evidence": {"1": [{"label": "SUPPORT"}]}},
            {"id": 11, "claim": "Second claim", "evidence": {"2": [{"label": "CONTRADICT"}], "3": [{"label": "SUPPORT"}]}},
            {"id": 12, "claim": "No evidence", "evidence": {}},
            {"id": 13, "claim": "Oversized evidence", "evidence": {"8": [{"label": "SUPPORT"}]}},
        ]

    def test_selection_is_order_independent_and_contains_all_gold_documents(self):
        documents, claims, counts = evaluate_scifact.select_dataset(self.corpus, self.claims, 2, 4)
        reversed_result = evaluate_scifact.select_dataset(list(reversed(self.corpus)), list(reversed(self.claims)), 2, 4)
        self.assertEqual((documents, claims, counts), reversed_result)
        self.assertEqual({claim["id"] for claim in claims}, {"10", "11"})
        self.assertTrue({"1", "2", "3"}.issubset({Path(key).stem for key, unused in documents}))
        self.assertEqual(counts["eligible_evidence_claims"], 2)
        self.assertEqual(counts["eligible_documents"], 7)
        self.assertEqual(counts["selected_evidence_documents"], 3)

    def test_selection_rejects_too_small_a_corpus(self):
        with self.assertRaises(ValueError):
            evaluate_scifact.select_dataset(self.corpus, self.claims, 2, 2)

    def test_lexical_pool_prioritizes_overlap_and_is_order_independent(self):
        documents = {"1": "gold relevant", "2": "gene therapy therapy", "3": "weather forecast", "4": "gene trial"}
        claims = [{"id": "1", "question": "gene therapy", "relevant": ["1"]}]
        pool = evaluate_scifact.lexical_pool(documents, claims, {"1"}, 2)
        self.assertEqual(pool, {"1", "2"})
        self.assertEqual(pool, evaluate_scifact.lexical_pool(dict(reversed(list(documents.items()))), claims, {"1"}, 2))

    def test_digest_changes_when_the_source_text_changes(self):
        documents, claims, unused = evaluate_scifact.select_dataset(self.corpus, self.claims, 2, 4)
        modified = list(documents)
        modified[0] = (modified[0][0], modified[0][1] + " changed")
        self.assertNotEqual(evaluate_scifact.corpus_digest(documents, claims),
                            evaluate_scifact.corpus_digest(modified, claims))

    def test_archive_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "data.tar.gz"
            archive.write_bytes(b"not the pinned dataset")
            with self.assertRaises(RuntimeError):
                evaluate_scifact.archive_path(archive)

    def test_metrics_use_all_annotated_documents_and_count_misses(self):
        observations = [
            {"mode": "keyword", "first_relevant_rank": 1, "returned_ids": ["2", "7"], "relevant": ["1", "2"], "query_ms": 10.0},
            {"mode": "keyword", "first_relevant_rank": None, "returned_ids": ["7"], "relevant": ["3"], "query_ms": 30.0},
        ]
        result = evaluate_scifact.summarize(observations)["keyword"]
        self.assertEqual(result["hit_at_1"], 1)
        self.assertEqual(result["hit_at_3"], 1)
        self.assertEqual(result["mrr_at_3"], 0.5)
        self.assertEqual(result["macro_recall_at_3"], 0.25)
        self.assertEqual(result["query_ms_median"], 20.0)
        self.assertEqual(result["query_ms_p95"], 30.0)

    def test_existing_vector_with_different_source_is_not_overwritten(self):
        vectors = Mock()
        key = core.vector_key("scifact/1.txt")
        vectors.get_vectors.return_value = {"vectors": [{"key": key, "metadata": {"source_version": "wrong"}}]}
        state = {"indexes": [evaluate_scifact.INDEX_NAME], "vector_bucket": "bucket"}
        with patch("scripts.evaluate_scifact.evaluate.check_index"), patch("scripts.evaluate_scifact.evaluate.index_keys", return_value={key}):
            with self.assertRaises(RuntimeError):
                evaluate_scifact.prepare_index(vectors, state, [("scifact/1.txt", "Example")], Mock())
        vectors.put_vectors.assert_not_called()

    def test_index_is_registered_for_cleanup_before_creation(self):
        vectors = Mock()
        state = {"indexes": ["documents"], "vector_bucket": "bucket"}
        missing = ClientError({"Error": {"Code": "NotFoundException", "Message": "missing"}}, "GetIndex")
        with patch("scripts.evaluate_scifact.lab.write_state") as save, \
             patch("scripts.evaluate_scifact.evaluate.check_index", side_effect=missing), \
             patch("scripts.evaluate_scifact.lab.create_index", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                evaluate_scifact.prepare_index(vectors, state, [], Mock())
        self.assertIn(evaluate_scifact.INDEX_NAME, state["indexes"])
        save.assert_called_once_with(state)


class LatencyProbeTests(unittest.TestCase):
    def test_latency_summary_separates_failures_from_success_percentiles(self):
        samples = [
            {"outcome": "success", "http_status": 200, "elapsed_ms": 10.0},
            {"outcome": "success", "http_status": 200, "elapsed_ms": 30.0},
            {"outcome": "http_error", "http_status": 502, "elapsed_ms": 1000.0},
        ]
        result = probe_latency.summarize(samples)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["success_p50_ms"], 20.0)
        self.assertEqual(result["success_p95_ms"], 30.0)
        self.assertEqual(result["http_statuses"], {"200": 2, "502": 1})
        self.assertEqual(result["http_responses"], 3)
        self.assertEqual(result["http_response_p50_ms"], 30.0)
        self.assertEqual(result["http_response_p95_ms"], 1000.0)

    def test_zero_successes_have_no_latency_percentiles(self):
        result = probe_latency.summarize([{"outcome": "client_error", "elapsed_ms": 1.0}])
        self.assertIsNone(result["success_p50_ms"])
        self.assertIsNone(result["success_p95_ms"])

    def test_timed_request_records_http_errors_without_private_payload(self):
        with patch("scripts.probe_latency.lab.request_api", return_value=(502, {"error": "private detail", "code": "ThrottlingException"})):
            result = probe_latency.timed_request(Mock(), {}, "hybrid")
        self.assertEqual(result["outcome"], "http_error")
        self.assertEqual(result["http_status"], 502)
        self.assertEqual(result["dependency_code"], "ThrottlingException")
        self.assertNotIn("private detail", json.dumps(result))

    def test_successful_http_response_can_still_miss_the_relevant_document(self):
        payload = {"mode": "hybrid", "results": []}
        with patch("scripts.probe_latency.lab.request_api", return_value=(200, payload)):
            result = probe_latency.timed_request(Mock(), {}, "hybrid")
        self.assertEqual(result["outcome"], "success")
        self.assertIsNone(result["first_relevant_rank"])
        summary = probe_latency.summarize([result])
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["retrieval_evaluated"], 1)
        self.assertEqual(summary["retrieval_hit_at_3"], 0)
        self.assertEqual(summary["retrieval_misses"], 1)

    def test_relevant_document_at_second_rank_is_not_a_hit_at_one(self):
        payload = {"mode": "hybrid", "results": [{"document_id": "other"}, {"document_id": "return301"}]}
        with patch("scripts.probe_latency.lab.request_api", return_value=(200, payload)):
            result = probe_latency.timed_request(Mock(), {}, "hybrid")
        summary = probe_latency.summarize([result])
        self.assertEqual(result["first_relevant_rank"], 2)
        self.assertEqual(summary["retrieval_hit_at_1"], 0)
        self.assertEqual(summary["retrieval_hit_at_3"], 1)


class SupportEvaluationTests(unittest.TestCase):
    def test_support_fixture_has_case_and_punctuation_lookalikes_and_negative_cases(self):
        documents, queries, missing, readiness, digest = evaluate_support.load_fixture()
        identifiers = {Path(key).stem for key, unused in documents}
        self.assertIn("AUTH-401", identifiers)
        self.assertIn("auth-401", identifiers)
        self.assertIn("AUTH.401", identifiers)
        self.assertIn("AUTH_401", identifiers)
        self.assertEqual(len(documents), 16)
        self.assertEqual(len(queries), 32)
        self.assertEqual(len(missing), 8)
        self.assertEqual(len(set(readiness.values())), len(documents))
        self.assertEqual(len(digest), 64)

    def test_fixture_refuses_false_missing_identifier(self):
        fixture = json.loads(evaluate_support.FIXTURE.read_text())
        fixture["missing_identifiers"] = [fixture["documents"][0]["id"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture))
            with self.assertRaises(ValueError):
                evaluate_support.load_fixture(path)

    def test_external_intent_selection_is_balanced_and_holds_out_rows(self):
        rows = [{"row": number, "question": f"Question {number}", "text": f"Response {number}",
                 "intent": "a" if number < 6 else "b"} for number in range(12)]
        documents, queries, readiness, count = evaluate_bitext.select_dataset(rows, 2, 2)
        reverse = evaluate_bitext.select_dataset(list(reversed(rows)), 2, 2)
        self.assertEqual((documents, queries, readiness, count), reverse)
        self.assertEqual((len(documents), len(queries), count), (4, 4, 2))
        selected_doc_rows = {int(Path(key).stem.removeprefix("bitext-")) for key, unused in documents}
        selected_query_rows = {int(query["id"].removeprefix("bitext-query-")) for query in queries}
        self.assertTrue(selected_doc_rows.isdisjoint(selected_query_rows))
        self.assertTrue(all(len(query["relevant"]) == 2 for query in queries))

    def test_external_dataset_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("wrong data")
            with self.assertRaises(RuntimeError):
                evaluate_bitext.dataset_path(path)


class PairedComparisonTests(unittest.TestCase):
    def test_counts_are_paired_and_resampling_is_reproducible(self):
        observations = [
            {"query_id": "one", "mode": "vector", "rank": 1, "relevant": ["a"]},
            {"query_id": "one", "mode": "hybrid", "rank": 2, "relevant": ["a"]},
            {"query_id": "two", "mode": "vector", "rank": None, "relevant": ["b"]},
            {"query_id": "two", "mode": "hybrid", "rank": 1, "relevant": ["b"]},
            {"query_id": "three", "mode": "vector", "rank": 1, "relevant": ["c"]},
            {"query_id": "three", "mode": "hybrid", "rank": 1, "relevant": ["c"]},
        ]
        first = compare_results.paired_summary(observations, resamples=200)
        self.assertEqual(first, compare_results.paired_summary(observations, resamples=200))
        self.assertEqual(first["paired_hit_at_1_counts"], {"both_hit": 1, "first_only": 1, "second_only": 1, "neither_hit": 0})
        self.assertEqual(first["second_minus_first_percentage_points"], 0)
        with self.assertRaises(ValueError):
            compare_results.paired_summary(observations[:-1], resamples=200)

    def test_differing_labels_are_rejected(self):
        observations = [
            {"claim_id": "one", "mode": "vector", "first_relevant_rank": 1, "relevant": ["a"]},
            {"claim_id": "one", "mode": "hybrid", "first_relevant_rank": 1, "relevant": ["b"]},
        ]
        with self.assertRaises(ValueError):
            compare_results.paired_summary(observations, resamples=200)


class RetryComparisonTests(unittest.TestCase):
    def test_retry_setting_preserves_other_variables_and_revision(self):
        functions = Mock()
        original = {"VECTOR_BUCKET": "bucket", "MODEL_MAX_ATTEMPTS": "3"}
        functions.get_function_configuration.return_value = {"Environment": {"Variables": original}, "RevisionId": "rev"}
        compare_retries.set_attempts(functions, "function", original, 1)
        functions.update_function_configuration.assert_called_once_with(
            FunctionName="function", Environment={"Variables": {"VECTOR_BUCKET": "bucket", "MODEL_MAX_ATTEMPTS": "1"}}, RevisionId="rev",
        )
        functions.get_waiter.return_value.wait.assert_called_once()

    def test_retry_setting_refuses_concurrent_environment_change(self):
        functions = Mock()
        functions.get_function_configuration.return_value = {"Environment": {"Variables": {"VECTOR_BUCKET": "different"}}, "RevisionId": "rev"}
        with self.assertRaises(RuntimeError):
            compare_retries.set_attempts(functions, "function", {"VECTOR_BUCKET": "bucket"}, 1)
        functions.update_function_configuration.assert_not_called()

    def test_restore_removes_retry_variable_if_it_was_originally_absent(self):
        functions = Mock()
        functions.get_function_configuration.return_value = {
            "Environment": {"Variables": {"VECTOR_BUCKET": "bucket", "MODEL_MAX_ATTEMPTS": "1"}}, "RevisionId": "rev",
        }
        compare_retries.set_attempts(functions, "function", {"VECTOR_BUCKET": "bucket"}, None)
        functions.update_function_configuration.assert_called_once_with(
            FunctionName="function", Environment={"Variables": {"VECTOR_BUCKET": "bucket"}}, RevisionId="rev",
        )

    def test_comparison_restores_environment_when_a_warmup_fails(self):
        session, functions = Mock(), Mock()
        original = {"MODEL_MAX_ATTEMPTS": "3"}
        arguments = SimpleNamespace(rounds_per_block=1, concurrency=3, include_logs=False, output="unused.json")
        with patch("scripts.compare_retries.lab.session_for_state", return_value=session), \
             patch("scripts.compare_retries.lab.load_state", return_value={}), \
             patch("scripts.compare_retries.verify_lab", return_value=(functions, "function", original)), \
             patch("scripts.compare_retries.set_attempts") as update, \
             patch("scripts.compare_retries.probe_latency.timed_request", return_value={"outcome": "http_error"}):
            with self.assertRaises(RuntimeError):
                compare_retries.run(arguments)
        self.assertEqual(update.call_args_list[0].args, (functions, "function", original, 1))
        self.assertEqual(update.call_args_list[-1].args, (functions, "function", original, None))

    def test_pooled_summary_uses_request_observations_and_rejects_mismatched_workloads(self):
        report = {field: "same" for field in summarize_retries.COMPARABLE_FIELDS}
        report.update({"concurrency": 1, "rounds_per_block": 1})
        sample = {"outcome": "success", "http_status": 200, "elapsed_ms": 10.0, "first_relevant_rank": 1}
        report["blocks"] = [{"model_total_max_attempts": attempts, "bursts": [{"samples": [sample]}]}
                            for attempts in (1, 3, 3, 1)]
        result = summarize_retries.summarize_reports([report, report])
        self.assertEqual(result["pooled"]["1"]["attempts"], 4)
        self.assertEqual(result["pooled"]["3"]["retrieval_hit_at_1"], 4)
        with self.assertRaises(ValueError):
            summarize_retries.summarize_reports([report, {**report, "region": "different"}])


class RecordedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def report(self, name):
        return json.loads((self.root / "results" / name).read_text())

    def test_support_observations_reproduce_the_saved_summary(self):
        unused_documents, unused_queries, unused_missing, unused_readiness, digest = evaluate_support.load_fixture()
        for name in ("2026-11-18-support-identifiers.json", "2026-11-18-console-support.json"):
            with self.subTest(report=name):
                report = self.report(name)
                self.assertEqual(report["fixture_sha256"], digest)
                self.assertEqual(report["summary"], evaluate.summarize(report["observations"]))
                positives = [item for item in report["exact_checks"] if item["expected"] is not None]
                negatives = [item for item in report["exact_checks"] if item["expected"] is None]
                self.assertEqual(report["exact_summary"], {
                    "positive_correct": sum(item["correct"] for item in positives), "positive_total": len(positives),
                    "negative_correct": sum(item["correct"] for item in negatives), "negative_total": len(negatives),
                })

    def test_screenshot_support_totals_match_the_figure(self):
        report = self.report("2026-11-18-console-support.json")
        for mode, expected in {"keyword": (22, 30), "vector": (21, 28), "hybrid": (24, 31)}.items():
            summary = report["summary"]["all"][mode]
            self.assertEqual((summary["queries"], summary["hit_at_1"], summary["hit_at_3"]), (32, *expected))
        self.assertEqual(report["exact_summary"], {
            "positive_correct": 16, "positive_total": 16, "negative_correct": 8, "negative_total": 8,
        })

    def test_screenshot_retry_summary_matches_recorded_requests(self):
        report = self.report("2026-11-18-console-retry.json")
        self.assertEqual([block["model_total_max_attempts"] for block in report["blocks"]], [1, 3, 3, 1])
        self.assertEqual(report["concurrency"], 3)
        self.assertEqual(report["rounds_per_block"], 3)
        for attempts, expected in {1: (11, 7), 3: (18, 0)}.items():
            samples = [sample for block in report["blocks"] if block["model_total_max_attempts"] == attempts
                       for burst in block["bursts"] for sample in burst["samples"]]
            self.assertEqual(len(samples), 18)
            summary = report["summary"][str(attempts)]
            self.assertEqual(summary, probe_latency.summarize(samples))
            self.assertEqual((summary["http_statuses"].get("200", 0), summary["http_statuses"].get("502", 0)), expected)
            self.assertEqual(summary["retrieval_hit_at_3"], expected[0])

    def test_screenshot_smoke_and_model_checks(self):
        smoke = self.report("2026-11-18-console-smoke.json")
        self.assertEqual(smoke["indexed_vectors_before_checks"], 16)
        self.assertEqual({name: check["http_status"] for name, check in smoke["checks"].items()}, {
            "anonymous_request": 403, "invalid_request": 400, "signed_keyword": 200,
            "signed_exact": 200, "signed_exact_case_sensitive": 200, "signed_exact_missing": 200,
            "signed_exact_invalid": 400, "sagemaker_vector": 200, "sagemaker_hybrid": 200,
        })
        self.assertTrue(all(check["passed"] for check in smoke["checks"].values()))
        model = self.report("2026-11-18-console-model.json")
        self.assertEqual({item["input_type"] for item in model["observations"]}, {"document", "query"})
        self.assertEqual(len(model["observations"]), 2)
        for item in model["observations"]:
            self.assertEqual(item["dimensions"], 384)
            self.assertAlmostEqual(item["l2_norm"], 1.0, places=4)

    def test_external_intent_results_reproduce_the_saved_summary(self):
        report = self.report("2026-11-18-bitext-support.json")
        self.assertEqual(report["dataset_sha256"], evaluate_bitext.DATASET_SHA256)
        self.assertEqual(report["dataset_revision"], evaluate_bitext.REVISION)
        self.assertEqual(report["summary"], evaluate_scifact.summarize(report["observations"]))

    def test_walkthrough_estimate_total_matches_itemized_assumptions(self):
        estimate = self.report("2026-11-18-walkthrough-estimate.json")
        total = sum(Decimal(item["estimated_charge_usd"]) for item in estimate["items"])
        self.assertEqual(total, Decimal(estimate["total_estimated_usd"]))
        self.assertIn("not measured", estimate["disclaimer"])

    def test_paired_results_match_their_source_hashes_and_observations(self):
        report = self.report("2026-11-18-paired-comparisons.json")
        for comparison in report["comparisons"]:
            path = self.root / "results" / comparison["report_file"]
            data = path.read_bytes()
            self.assertEqual(comparison["report_sha256"], hashlib.sha256(data).hexdigest())
            calculated = compare_results.paired_summary(json.loads(data)["observations"],
                                                        comparison["first_mode"], comparison["second_mode"],
                                                        comparison["seed"], comparison["resamples"])
            self.assertEqual(calculated, {key: comparison[key] for key in calculated})

    def test_pooled_retry_results_match_the_raw_requests(self):
        report = self.report("2026-11-18-retry-pooled.json")
        sources = []
        for source in report["sources"]:
            data = (self.root / "results" / source["report_file"]).read_bytes()
            self.assertEqual(source["report_sha256"], hashlib.sha256(data).hexdigest())
            sources.append(json.loads(data))
        self.assertEqual(report["pooled"], summarize_retries.summarize_reports(sources)["pooled"])


class DeploymentPreflightTests(unittest.TestCase):
    def test_old_sdk_is_rejected_before_cloud_resources_are_created(self):
        session = Mock()
        session.client.return_value.meta.service_model.operation_names = ["QueryVectors"]
        with patch("scripts.lab.STATE_FILE") as state_file, patch("scripts.lab.boto3.Session", return_value=session), \
             patch("scripts.lab.write_state") as save, patch("scripts.lab.check_serverless_quota") as quotas:
            state_file.exists.return_value = False
            with self.assertRaisesRegex(RuntimeError, "too old"):
                lab.deploy(SimpleNamespace(region="us-west-2"))
        save.assert_not_called()
        quotas.assert_not_called()
        session.client.assert_called_once_with("s3vectors")


class ReleaseTests(unittest.TestCase):
    def test_release_is_reproducible_and_excludes_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "VERSION").write_text("1.0.0\n")
            (root / "README.md").write_text("Example\n")
            (root / ".env").write_text("private\n")
            (root / "src").mkdir()
            (root / "src" / "example.py").write_text("VALUE = 1\n")
            with patch("scripts.create_release.TOP_LEVEL", ("VERSION", "README.md")), \
                 patch("scripts.create_release.DIRECTORIES", ("src",)), patch("scripts.create_release.RESULTS", ()):
                path, first_digest = create_release.create_release(root)
                first = path.read_bytes()
                unused_path, second_digest = create_release.create_release(root)
            self.assertEqual(first_digest, second_digest)
            self.assertEqual(first, path.read_bytes())
            with zipfile.ZipFile(path) as archive:
                self.assertNotIn(".env", archive.namelist())
                manifest = json.loads(archive.read("RELEASE-MANIFEST.json"))
                self.assertEqual(manifest["version"], "1.0.0")
                self.assertEqual({entry["path"] for entry in manifest["files"]}, {"VERSION", "README.md", "src/example.py"})

    def test_release_refuses_an_access_key_in_included_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "VERSION").write_text("1.0.0\n")
            (root / "README.md").write_text("AKIA" + "A" * 16)
            (root / "src").mkdir()
            with patch("scripts.create_release.TOP_LEVEL", ("VERSION", "README.md")), \
                 patch("scripts.create_release.DIRECTORIES", ("src",)), patch("scripts.create_release.RESULTS", ()):
                with self.assertRaises(RuntimeError):
                    create_release.create_release(root)

    def test_release_verifier_rejects_bad_hashes_and_missing_links(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.zip"
            manifest = {"version": "1.0.0", "files": [{"path": "VERSION", "sha256": "0" * 64}]}
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("VERSION", "1.0.0\n")
                archive.writestr("RELEASE-MANIFEST.json", json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                verify_release.verify_archive(path)
            root = Path(directory)
            (root / "VERSION").write_text("1.0.0\n")
            (root / "README.md").write_text("[missing](missing.txt)\n")
            (root / "src").mkdir()
            (root / "src" / "example.py").write_text("VALUE = 1\n")
            with patch("scripts.create_release.TOP_LEVEL", ("VERSION", "README.md")), \
                 patch("scripts.create_release.DIRECTORIES", ("src",)), patch("scripts.create_release.RESULTS", ()):
                release, unused_digest = create_release.create_release(root)
            with self.assertRaisesRegex(RuntimeError, "Missing relative link"):
                verify_release.verify_archive(release)


if __name__ == "__main__":
    unittest.main()
