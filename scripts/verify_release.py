#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import posixpath
import re
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from pathlib import Path, PurePosixPath


PRIVATE_PARTS = {".env", ".lab-state.json", ".build", ".venv", "dist", "__pycache__"}
LINK_PATTERN = re.compile(r"\]\(([^)]+)\)")
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


def verify_archive(path, run_tests=False):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [entry.filename for entry in infos]
        if len(set(names)) != len(names) or sum(entry.file_size for entry in infos) > MAX_ARCHIVE_BYTES:
            raise RuntimeError("Archive contains duplicate entries or exceeds the size limit.")
        for entry in infos:
            name = entry.filename
            parts = PurePosixPath(name).parts
            if (not name or name.startswith("/") or "\\" in name or ".." in parts
                    or re.match(r"^[A-Za-z]:", name) or str(PurePosixPath(name)) != name
                    or any(part in PRIVATE_PARTS or part.startswith(".env.") for part in parts)
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise RuntimeError(f"Archive contains an unsafe or private path: {name}")
        manifest = json.loads(archive.read("RELEASE-MANIFEST.json"))
        entries = manifest["files"]
        listed = {entry["path"]: entry["sha256"] for entry in entries}
        if len(listed) != len(entries) or set(listed) != set(names) - {"RELEASE-MANIFEST.json"}:
            raise RuntimeError("Archive contents do not match its manifest.")
        for name, expected in listed.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError(f"Archive checksum mismatch: {name}")
        if archive.read("VERSION").decode().strip() != manifest["version"]:
            raise RuntimeError("Archive version does not match the manifest.")
        link_count = 0
        for document in ("ARTICLE.md", "BENCHMARKS.md", "README.md", "SCREENSHOTS.md"):
            if document not in listed:
                continue
            text = archive.read(document).decode("utf-8")
            for match in LINK_PATTERN.finditer(text):
                target = match.group(1).strip().split()[0]
                parsed = urllib.parse.urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(document), urllib.parse.unquote(parsed.path)))
                if resolved.startswith("../") or resolved.startswith("/"):
                    raise RuntimeError(f"Relative link escapes the archive: {document} -> {target}")
                if resolved not in listed and not any(name.startswith(resolved.rstrip("/") + "/") for name in listed):
                    raise RuntimeError(f"Missing relative link in archive: {document} -> {target}")
                link_count += 1
        if run_tests:
            with tempfile.TemporaryDirectory(prefix="hybrid-search-release-") as temporary:
                archive.extractall(temporary)
                environment = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
                environment["PYTHONPATH"] = os.pathsep.join((str(Path(temporary) / "src"), temporary))
                environment["PYTHONNOUSERSITE"] = "1"
                environment["AWS_EC2_METADATA_DISABLED"] = "true"
                environment["AWS_SHARED_CREDENTIALS_FILE"] = str(Path(temporary) / "no-credentials")
                environment["AWS_CONFIG_FILE"] = str(Path(temporary) / "no-config")
                subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                               cwd=temporary, env=environment, check=True)
        return {"version": manifest["version"], "files": len(listed), "relative_links_checked": link_count,
                "clean_extraction_tests": "passed" if run_tests else "not run"}


def main():
    parser = argparse.ArgumentParser(description="Verify an archive manifest, relative links, and optionally tests in a clean extraction.")
    parser.add_argument("archive")
    parser.add_argument("--run-tests", action="store_true")
    arguments = parser.parse_args()
    try:
        print(json.dumps(verify_archive(arguments.archive, arguments.run_tests), indent=2))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
