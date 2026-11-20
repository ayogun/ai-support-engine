#!/usr/bin/env python3
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ("VERSION", "README.md", "LICENSE", ".gitignore",
             "requirements.txt", "requirements-eval.txt", "template.yaml")
DIRECTORIES = ("assets", "checks", "demo", "fixtures", "scripts", "src", "tests")
RESULTS = (
    "bge-evaluation.json", "ingest-observation.json", "live-smoke.json", "titan-quotas-us-west-2.json",
    "model-preflight.json", "sagemaker-model-check.json", "sagemaker-smoke.json",
    "sagemaker-ingest-observation.json", "sagemaker-evaluation.json",
    "2026-11-18-model-check.json", "2026-11-18-smoke.json", "2026-11-18-infrastructure.json",
    "2026-11-18-burst-diagnostics.json", "2026-11-18-cost-example.json", "2026-11-18-walkthrough-estimate.json",
    "2026-11-18-latency.json", "2026-11-18-latency-repeat.json", "2026-11-18-synthetic.json",
    "2026-11-18-scifact.json", "2026-11-18-scifact-lexical.json", "2026-11-18-scifact-repeat.json",
    "2026-11-18-api-response.json", "2026-11-18-post-update-smoke.json",
    "2026-11-18-latency-instrumented.json", "2026-11-18-instrumented-diagnostics.json",
    "2026-11-18-cleanup.json",
    "2026-11-18-revision-model-check.json", "2026-11-18-revision-smoke.json",
    "2026-11-18-revision-infrastructure.json", "2026-11-18-revision-cleanup.json",
    "2026-11-18-support-identifiers.json", "2026-11-18-bitext-support.json",
    "2026-11-18-techqa-support.json", "2026-11-18-techqa-paired.json", "2026-11-18-techqa-cleanup.json",
    "2026-11-18-paired-historical.json", "2026-11-18-paired-support.json", "2026-11-18-paired-comparisons.json",
    "2026-11-18-retry-comparison.json", "2026-11-18-retry-repeat.json", "2026-11-18-retry-pooled.json",
    "2026-11-18-screenshot-infrastructure.json", "2026-11-18-screenshot-model.json",
    "2026-11-18-screenshot-smoke.json", "2026-11-18-screenshot-support.json",
    "2026-11-18-screenshot-retry.json", "2026-11-18-screenshot-cleanup.json",
    "2026-11-18-console-infrastructure.json", "2026-11-18-console-model.json",
    "2026-11-18-console-smoke.json", "2026-11-18-console-support.json",
    "2026-11-18-console-retry.json", "2026-11-18-console-cleanup.json",
)
ALLOWED_SUFFIXES = {".py", ".png", ".guard", ".json", ".txt"}
EXTRA_ASSETS = {"assets/architecture.drawio", "assets/architecture.md",
                "demo/index.html", "demo/styles.css", "demo/app.js"}
ACCESS_KEY = re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")


def collect_files(root):
    files = [root / name for name in TOP_LEVEL]
    files.extend(root / "results" / name for name in RESULTS)
    for name in DIRECTORIES:
        directory = root / name
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(f"Missing or unsafe release directory: {name}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"Refusing a symbolic link in the release: {path.relative_to(root)}")
            relative = path.relative_to(root)
            if (path.is_file() and (path.suffix in ALLOWED_SUFFIXES or relative.as_posix() in EXTRA_ASSETS)
                    and not any(part.startswith(".") for part in relative.parts)):
                files.append(path)
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing or unsafe release file: {path.relative_to(root)}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def zip_entry(name, data):
    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info, data


def create_release(root):
    version = (root / "VERSION").read_text().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError("VERSION must be a semantic version such as 1.0.0.")
    files = collect_files(root)
    entries = []
    for path in files:
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if ACCESS_KEY.search(data):
            raise RuntimeError(f"Possible AWS access key in {name}; refusing release.")
        entries.append((name, data))
    manifest = {
        "version": version,
        "files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in entries],
    }
    entries.append(("RELEASE-MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode()))
    destination = root / "dist" / f"serverless-hybrid-search-v{version}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in entries:
                info, payload = zip_entry(name, data)
                archive.writestr(info, payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(destination.read_bytes()).hexdigest()


def main():
    try:
        path, digest = create_release(ROOT)
        print("Created", path.relative_to(ROOT))
        print("SHA-256:", digest)
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Release creation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
