import io
import json
import os
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import create_release


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureDiagramTests(unittest.TestCase):
    def test_native_aws_icons_and_connected_paths(self):
        source = (ROOT / "assets" / "architecture.drawio").read_text()
        self.assertNotIn("<!--", source)
        cells = ET.fromstring(source).findall(".//mxCell")
        cells_by_id = {cell.get("id"): cell for cell in cells}
        self.assertEqual(len(cells), len(cells_by_id))
        icons = {
            "document-bucket": "resIcon=mxgraph.aws4.s3;",
            "ingest-lambda": "resIcon=mxgraph.aws4.lambda;",
            "failed-queue": "resIcon=mxgraph.aws4.sqs;",
            "ingest-model": "resIcon=mxgraph.aws4.sagemaker;",
            "ingest-vectors": "shape=mxgraph.aws4.s3_vectors;",
            "api-gateway": "resIcon=mxgraph.aws4.api_gateway;",
            "search-lambda": "resIcon=mxgraph.aws4.lambda;",
            "search-model": "resIcon=mxgraph.aws4.sagemaker;",
            "search-vectors": "shape=mxgraph.aws4.s3_vectors;",
        }
        for cell_id, icon in icons.items():
            with self.subTest(cell_id=cell_id):
                self.assertIn(icon, cells_by_id[cell_id].get("style", ""))
        expected_paths = {
            ("uploader", "document-bucket"), ("document-bucket", "ingest-lambda"),
            ("ingest-lambda", "ingest-vectors"), ("ingest-lambda", "ingest-model"),
            ("ingest-lambda", "failed-queue"), ("client", "api-gateway"),
            ("api-gateway", "search-lambda"), ("search-lambda", "search-vectors"),
            ("search-lambda", "search-model"),
        }
        edges = [cell for cell in cells if cell.get("edge") == "1"]
        self.assertEqual({(edge.get("source"), edge.get("target")) for edge in edges}, expected_paths)
        for edge in edges:
            self.assertIn(edge.get("source"), cells_by_id)
            self.assertIn(edge.get("target"), cells_by_id)
            self.assertEqual(edge.find("mxGeometry").get("relative"), "1")

    def test_release_includes_editable_diagram_and_guide(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            for name in ("architecture.drawio", "architecture.md", "private.md"):
                (assets / name).write_text("example")
            with patch.object(create_release, "TOP_LEVEL", ()), patch.object(create_release, "RESULTS", ()), \
                    patch.object(create_release, "DIRECTORIES", ("assets",)):
                paths = {path.relative_to(root).as_posix() for path in create_release.collect_files(root)}
        self.assertEqual(paths, {"assets/architecture.drawio", "assets/architecture.md"})


class SignedRequestExampleTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "ARTICLE.md").is_file(), "The article is published separately")
    def test_article_example_signs_the_expected_json_and_session_token(self):
        article = (Path(__file__).resolve().parents[1] / "ARTICLE.md").read_text()
        examples = re.findall(r"python - <<'PY'\n(.*?)\nPY", article, flags=re.DOTALL)
        matches = [example for example in examples if "SigV4Auth" in example]
        self.assertEqual(len(matches), 1)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"mode":"hybrid","results":[]}'
        environment = {
            "SEARCH_URL": "https://example.execute-api.us-west-2.amazonaws.com/search",
            "AWS_REGION": "us-west-2", "AWS_ACCESS_KEY_ID": "example-access",
            "AWS_SECRET_ACCESS_KEY": "example-secret", "AWS_SESSION_TOKEN": "example-session",
            "AWS_EC2_METADATA_DISABLED": "true",
        }
        with patch.dict(os.environ, environment, clear=True), patch("urllib.request.urlopen", return_value=response) as urlopen:
            with redirect_stdout(io.StringIO()):
                exec(compile(matches[0], "ARTICLE.md", "exec"), {})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {"question": "Find RETURN301", "mode": "hybrid", "limit": 3})
        self.assertIn("Credential=example-access/", request.get_header("Authorization"))
        self.assertEqual(request.get_header("X-amz-security-token"), "example-session")


if __name__ == "__main__":
    unittest.main()
