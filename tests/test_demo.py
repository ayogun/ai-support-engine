import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

from demo import server as demo_server
from scripts import create_release


ROOT = Path(__file__).resolve().parents[1]


class DemoDataTests(unittest.TestCase):
    def test_replays_the_saved_rankings_and_exact_lookups(self):
        data = demo_server.load_demo_data()
        fixture = json.loads(demo_server.FIXTURE.read_text())
        report = json.loads(demo_server.REPORT.read_text())
        saved = {(row["query_id"], row["mode"]): row for row in report["observations"]}
        self.assertEqual(len(data["queries"]), len(fixture["queries"]))
        self.assertEqual(len(data["documents"]), 16)
        for query in data["queries"]:
            for mode in demo_server.MODES:
                row = saved[(query["id"], mode)]
                self.assertEqual(query["modes"][mode]["returned_ids"], row["returned_ids"])
                self.assertEqual(query["modes"][mode]["rank"], row["rank"])
        self.assertEqual(data["exact_checks"]["AUTH-401"], ["AUTH-401"])
        self.assertEqual(data["exact_checks"]["Auth-401"], [])
        example = next(query for query in data["queries"] if query["id"] == "identifier-01")
        self.assertEqual(example["modes"]["keyword"]["rank"], 3)
        self.assertIsNone(example["modes"]["hybrid"]["rank"])
        self.assertEqual(example["modes"]["hybrid"]["returned_ids"], ["auth-401", "AUTH.401", "AUTH_401"])
        paraphrase = next(query for query in data["queries"] if query["id"] == "support-14")
        self.assertEqual(paraphrase["relevant"], ["API_E500"])
        self.assertIsNone(paraphrase["modes"]["keyword"]["rank"])
        self.assertEqual(paraphrase["modes"]["vector"]["rank"], 1)
        self.assertEqual(paraphrase["modes"]["hybrid"]["rank"], 2)

    def test_changed_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.json"
            fixture.write_text(demo_server.FIXTURE.read_text() + " ")
            with patch.object(demo_server, "FIXTURE", fixture):
                with self.assertRaisesRegex(ValueError, "no longer matches"):
                    demo_server.load_demo_data()

    def test_demo_files_are_included_in_release(self):
        paths = {path.relative_to(ROOT).as_posix() for path in create_release.collect_files(ROOT)}
        self.assertTrue({"demo/server.py", "demo/index.html", "demo/styles.css", "demo/app.js",
                         "assets/support-search-demo.png", "assets/support-search-exact-test.png",
                         "assets/support-search-vector-test.png"} <= paths)
        self.assertTrue(paths.isdisjoint({"ARTICLE.md", "BENCHMARKS.md"}))

    @unittest.skipUnless((ROOT / "ARTICLE.md").is_file(), "The article is published separately")
    def test_article_labels_the_screenshot_as_an_offline_illustration(self):
        article = (ROOT / "ARTICLE.md").read_text()
        self.assertIn("assets/support-search-demo.png", article)
        self.assertIn("not a deployed interface or a live AWS request", article)
        self.assertNotIn("\N{EM DASH}", article)

    @unittest.skipUnless((ROOT / "ARTICLE.md").is_file(), "The article is published separately")
    def test_result_screenshots_are_pngs_with_offline_captions(self):
        article = (ROOT / "ARTICLE.md").read_text()
        for path in ("assets/support-search-exact-test.png", "assets/support-search-vector-test.png"):
            with self.subTest(path=path):
                self.assertEqual((ROOT / path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                marker = f"]({path})\n\n"
                self.assertIn(marker, article)
                caption = article.split(marker, 1)[1].splitlines()[0]
                self.assertIn("Offline replay", caption)
                self.assertIn("not a live API request", caption)
                self.assertIn("or a deployed GUI", caption)


class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = demo_server.create_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def test_only_local_address_and_allowed_assets_are_served(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        for path, content_type in (("/", "text/html"), ("/styles.css", "text/css"),
                                   ("/app.js", "text/javascript"), ("/demo-data.json", "application/json")):
            with self.subTest(path=path), urlopen(self.base_url + path, timeout=3) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.headers["Content-Type"].startswith(content_type))
                self.assertIn("connect-src 'self'", response.headers["Content-Security-Policy"])
        for path in ("/.env", "/../.env", "/%2e%2e/.env", "/server.py"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                urlopen(self.base_url + path, timeout=3)
            self.assertEqual(error.exception.code, 404)

    def test_data_endpoint_exposes_saved_synthetic_examples(self):
        with urlopen(self.base_url + "/demo-data.json", timeout=3) as response:
            data = json.load(response)
        self.assertEqual(data["queries"][0]["question"], "Find AUTH-401")
        self.assertEqual(data["top_k"], 3)
        self.assertEqual(data["exact_checks"]["AUTH-401"], ["AUTH-401"])


if __name__ == "__main__":
    unittest.main()
