#!/usr/bin/env python3
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "support_cases.json"
REPORT = ROOT / "results" / "2026-11-18-support-identifiers.json"
MODES = ("keyword", "vector", "hybrid")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def load_demo_data():
    fixture_bytes = FIXTURE.read_bytes()
    fixture = json.loads(fixture_bytes)
    report = json.loads(REPORT.read_text())
    if hashlib.sha256(fixture_bytes).hexdigest() != report["fixture_sha256"]:
        raise ValueError("The support fixture no longer matches the recorded evaluation.")

    documents = {document["id"]: document["text"] for document in fixture["documents"]}
    observations = {}
    for row in report["observations"]:
        key = (row["query_id"], row["mode"])
        if key in observations:
            raise ValueError(f"Duplicate recorded observation: {key}")
        observations[key] = row

    queries = []
    for query in fixture["queries"]:
        modes = {}
        for mode in MODES:
            row = observations[(query["id"], mode)]
            returned_ids = row["returned_ids"]
            if any(document_id not in documents for document_id in returned_ids):
                raise ValueError(f"Unknown document in {query['id']} / {mode}.")
            expected_rank = next((index + 1 for index, document_id in enumerate(returned_ids)
                                  if document_id in query["relevant"]), None)
            if row["rank"] != expected_rank:
                raise ValueError(f"Recorded rank does not match {query['id']} / {mode}.")
            modes[mode] = {"returned_ids": returned_ids, "rank": row["rank"]}
        queries.append({
            "id": query["id"],
            "kind": query["kind"],
            "question": query["question"],
            "known_id": query.get("identifier", ""),
            "relevant": query["relevant"],
            "modes": modes,
        })

    exact_checks = {}
    for check in report["exact_checks"]:
        identifier = check["identifier"]
        if identifier in exact_checks or any(item not in documents for item in check["returned_ids"]):
            raise ValueError(f"Invalid recorded exact lookup: {identifier}")
        exact_checks[identifier] = check["returned_ids"]

    return {
        "recorded_at": report["timestamp_utc"],
        "top_k": report["top_k"],
        "documents": documents,
        "queries": queries,
        "exact_checks": exact_checks,
    }


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/demo-data.json":
            payload = json.dumps(self.server.demo_data, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            payload = (DEMO / filename).read_bytes()
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; "
                         "frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_string, *args):
        return


def create_server(port=8765):
    data = load_demo_data()
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)
    server.demo_data = data
    return server


def main():
    parser = argparse.ArgumentParser(description="Preview saved synthetic support-search results locally.")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")
    arguments = parser.parse_args()
    server = create_server(arguments.port)
    print(f"Offline demo: http://127.0.0.1:{server.server_port}/", flush=True)
    print("Only saved synthetic results are served. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
