import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import compare_results, evaluate_scifact, evaluate_techqa


class TechQASelectionTests(unittest.TestCase):
    def setUp(self):
        self.documents = {str(number): f"Document {number}" for number in range(1, 8)}
        self.rows = [
            {"id": "DEV_Q001", "is_impossible": False, "question": "First question",
             "contexts": [{"filename": "1.txt", "text": "Document 1"}]},
            {"id": "DEV_Q002", "is_impossible": False, "question": "Second question",
             "contexts": [{"filename": "2.txt", "text": "Document 2"}]},
            {"id": "DEV_Q003", "is_impossible": False, "question": "Same document",
             "contexts": [{"filename": "1.txt", "text": "Document 1"}]},
            {"id": "DEV_Q004", "is_impossible": True, "question": "Unanswerable", "contexts": []},
            {"id": "DEV_Q005", "is_impossible": False, "question": "Mismatched context",
             "contexts": [{"filename": "3.txt", "text": "Different text"}]},
            {"id": "TRAIN_Q001", "is_impossible": False, "question": "Training item",
             "contexts": [{"filename": "4.txt", "text": "Document 4"}]},
        ]

    def test_selection_is_stable_and_forces_distinct_linked_documents(self):
        selected = evaluate_techqa.select_dataset(self.documents, self.rows, 7, 2, 4)
        reversed_input = evaluate_techqa.select_dataset(dict(reversed(list(self.documents.items()))), list(reversed(self.rows)), 7, 2, 4)
        self.assertEqual(selected, reversed_input)
        documents, questions, readiness, counts = selected
        self.assertEqual({query["relevant"][0] for query in questions}, {"1", "2"})
        self.assertTrue({"1", "2"}.issubset({Path(key).stem for key, unused in documents}))
        self.assertEqual(len(readiness), 4)
        self.assertEqual(counts["development_unanswerable_excluded"], 1)
        self.assertEqual(counts["eligible_answerable_questions"], 3)

    def test_rejects_insufficient_distinct_documents(self):
        with self.assertRaises(ValueError):
            evaluate_techqa.select_dataset(self.documents, self.rows, 7, 3, 4)
        with self.assertRaises(ValueError):
            evaluate_techqa.select_dataset(self.documents, self.rows, 7, 2, 1)

    def test_rejects_duplicate_or_ambiguous_labels(self):
        with self.assertRaises(ValueError):
            evaluate_techqa.select_dataset(self.documents, self.rows + [self.rows[0]], 7, 2, 4)
        broken = [dict(row) for row in self.rows]
        broken[0]["is_impossible"] = "False"
        with self.assertRaises(ValueError):
            evaluate_techqa.select_dataset(self.documents, broken, 7, 2, 4)

    def test_rejects_a_modified_cached_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "train.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                evaluate_techqa.dataset_path(directory, "train.json")

    def test_recorded_results_and_paired_comparison_are_recomputable(self):
        root = Path(__file__).resolve().parents[1]
        data = (root / "results" / "2026-11-18-techqa-support.json").read_bytes()
        report = json.loads(data)
        comparison = json.loads((root / "results" / "2026-11-18-techqa-paired.json").read_text())["comparisons"][0]
        self.assertEqual(report["dataset_revision"], evaluate_techqa.REVISION)
        self.assertEqual(report["source_sha256"], {name: details[0] for name, details in evaluate_techqa.FILES.items()})
        self.assertEqual((report["document_count"], report["query_count"]), (600, 100))
        self.assertEqual(report["summary"], evaluate_scifact.summarize(report["observations"]))
        self.assertEqual({mode: row["hit_at_1"] for mode, row in report["summary"].items()},
                         {"keyword": 85, "vector": 84, "hybrid": 85})
        self.assertTrue(all("question" not in item and "text" not in item for item in report["observations"]))
        self.assertEqual(comparison["report_sha256"], hashlib.sha256(data).hexdigest())
        recomputed = compare_results.paired_summary(report["observations"], comparison["first_mode"],
                                                   comparison["second_mode"], comparison["seed"], comparison["resamples"])
        self.assertEqual(recomputed, {key: comparison[key] for key in recomputed})


if __name__ == "__main__":
    unittest.main()
