#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
INK = "#172943"
MUTED = "#50627a"
BACKGROUND = "#f4f7fb"
WHITE = "#ffffff"
BLUE = "#2353a2"
CYAN = "#087e8b"
ORANGE = "#d06a18"
RED = "#a32938"


def font(size, bold=False):
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def text(draw, position, value, size=24, fill=INK, bold=False):
    draw.text(position, value, font=font(size, bold), fill=fill)


def panel(draw, box, fill=WHITE, outline="#d9e2ef", radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def results():
    report = json.loads((ROOT / "results" / "sagemaker-evaluation.json").read_text())
    image = Image.new("RGB", (1600, 960), BACKGROUND)
    draw = ImageDraw.Draw(image)
    text(draw, (70, 45), "What did hybrid retrieval actually improve?", 42, bold=True)
    text(draw, (70, 103), "Real S3 Vectors queries • SageMaker Serverless BGE embeddings", 24, MUTED)
    date = datetime.fromisoformat(report["timestamp_utc"]).strftime("%d %b %Y")
    text(draw, (70, 140), f'{report["document_count"]} synthetic documents · {report["query_count"]} labeled questions · {report["region"]} · {date}', 21, MUTED)
    colors = {"keyword": BLUE, "vector": CYAN, "hybrid": ORANGE}
    groups = [("exact_identifier", "Exact identifiers"), ("paraphrase", "Meaning / paraphrase")]
    for index, (group, title) in enumerate(groups):
        left = 70 + index * 750
        panel(draw, (left, 205, left + 710, 665))
        text(draw, (left + 30, 230), title, 30, INK, True)
        text(draw, (left + 30, 276), "Relevant document ranked #1", 20, MUTED)
        for offset, mode in enumerate(("keyword", "vector", "hybrid")):
            item = report["summary"][group][mode]
            top = 335 + offset * 94
            text(draw, (left + 30, top), mode.title(), 23, INK, True)
            draw.rounded_rectangle((left + 180, top + 5, left + 565, top + 37), radius=10, fill="#e6edf6")
            width = round(385 * item["hit_at_1"] / item["queries"])
            draw.rounded_rectangle((left + 180, top + 5, left + 180 + width, top + 37), radius=10, fill=colors[mode])
            text(draw, (left + 580, top), f'{item["hit_at_1"]}/{item["queries"]}', 24, INK, True)
    panel(draw, (70, 710, 1530, 850), "#fff8ed", "#f0d7b4")
    by_query = {}
    for item in report["observations"]:
        by_query.setdefault(item["query_id"], {})[item["mode"]] = item
    fixes = sum(modes["vector"]["rank"] != 1 and modes["hybrid"]["rank"] == 1 for modes in by_query.values() if modes["vector"]["kind"] == "exact_identifier")
    regressions = sum(modes["vector"]["rank"] == 1 and modes["hybrid"]["rank"] != 1 for modes in by_query.values() if modes["vector"]["kind"] == "paraphrase")
    text(draw, (100, 735), f"Hybrid fixed {fixes} identifier misses, but ranked {regressions} vector-first paraphrases below #1.", 24, INK, True)
    text(draw, (100, 783), "Small, hand-labeled toy corpus: directional evidence only; not a production benchmark.", 22, MUTED)
    text(draw, (70, 895), "Generated from results/sagemaker-evaluation.json; not a screenshot or a service benchmark.", 20, MUTED)
    image.save(ASSETS / "retrieval-results.png", optimize=True)


def live_checks():
    report = json.loads((ROOT / "results" / "sagemaker-smoke.json").read_text())
    ingest = json.loads((ROOT / "results" / "sagemaker-ingest-observation.json").read_text())
    image = Image.new("RGB", (1600, 840), BACKGROUND)
    draw = ImageDraw.Draw(image)
    text(draw, (70, 45), "Live AWS wiring: what the tests verified", 42, bold=True)
    text(draw, (70, 103), "Generated from saved AWS test results—not an AWS console screenshot", 23, MUTED)
    panel(draw, (70, 165, 1530, 740))
    vector = report["checks"]["sagemaker_vector"]
    hybrid = report["checks"]["sagemaker_hybrid"]
    rows = [
        ("Unsigned HTTP request", str(report["checks"]["anonymous_request"]["http_status"]), "Rejected as intended" if report["checks"]["anonymous_request"]["passed"] else "Unexpected authorization result", CYAN if report["checks"]["anonymous_request"]["passed"] else RED),
        ("Signed invalid request", str(report["checks"]["invalid_request"]["http_status"]), "Validated as intended" if report["checks"]["invalid_request"]["passed"] else "Unexpected validation result", CYAN if report["checks"]["invalid_request"]["passed"] else RED),
        ("Indexed documents", str(report["indexed_vectors_before_checks"]), "Actual vectors present before the API checks", CYAN),
        ("Signed keyword search", str(report["checks"]["signed_keyword"]["http_status"]), "Keyword request succeeded" if report["checks"]["signed_keyword"]["passed"] else "Keyword request failed", CYAN if report["checks"]["signed_keyword"]["passed"] else RED),
        ("Signed vector search", str(vector["http_status"]), "SageMaker-backed vector request succeeded" if vector["passed"] else "Vector request failed", CYAN if vector["passed"] else RED),
        ("Signed hybrid search", str(hybrid["http_status"]), "SageMaker-backed hybrid request succeeded" if hybrid["passed"] else "Hybrid request failed", CYAN if hybrid["passed"] else RED),
        ("Failed ingestion queue", str(ingest["approximate_visible_dlq_messages"]), "Retained earlier IAM failure; inspect before replay", ORANGE),
    ]
    for index, (label, value, detail, color) in enumerate(rows):
        row = 190 + index * 75
        text(draw, (110, row), label, 25, INK, True)
        text(draw, (610, row), value, 28, color, True)
        text(draw, (735, row + 3), detail, 21, MUTED)
        if index < len(rows) - 1:
            draw.line((110, row + 58, 1490, row + 58), fill="#e4eaf2", width=2)
    text(draw, (70, 780), "The queue message came from an IAM error fixed before the successful ingestion and search checks.", 20, MUTED)
    image.save(ASSETS / "live-checks.png", optimize=True)


def scifact_comparison():
    pools = [
        ("Hash-sampled pool", json.loads((ROOT / "results" / "2026-11-18-scifact.json").read_text())),
        ("Lexical challenge pool", json.loads((ROOT / "results" / "2026-11-18-scifact-lexical.json").read_text())),
    ]
    image = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    text(draw, (70, 45), "Same claims, different candidate pools", 42, bold=True)
    text(draw, (70, 105), "SciFact development subset · 50 claims · 300 abstracts in each pool", 24, MUTED)
    colors = {"keyword": BLUE, "vector": CYAN, "hybrid": ORANGE}
    for index, (title, report) in enumerate(pools):
        left = 70 + index * 750
        panel(draw, (left, 180, left + 710, 650))
        text(draw, (left + 30, 210), title, 30, INK, True)
        text(draw, (left + 30, 258), "Known evidence ranked #1", 21, MUTED)
        for offset, mode in enumerate(("keyword", "vector", "hybrid")):
            item = report["summary"][mode]
            top = 322 + offset * 94
            text(draw, (left + 30, top), mode.title(), 23, INK, True)
            draw.rounded_rectangle((left + 180, top + 5, left + 565, top + 37), radius=10, fill="#e6edf6")
            width = round(385 * item["hit_at_1"] / item["queries"])
            if width:
                draw.rounded_rectangle((left + 180, top + 5, left + 180 + width, top + 37), radius=10, fill=colors[mode])
            text(draw, (left + 580, top), f'{item["hit_at_1"]}/{item["queries"]}', 24, INK, True)
    panel(draw, (70, 700, 1530, 815), "#fff8ed", "#f0d7b4")
    text(draw, (100, 720), "Both pools force in the known evidence; the lexical pool deliberately favors overlapping terms.", 22, INK, True)
    text(draw, (100, 759), "Unjudged documents may be relevant. These are small, selected comparisons—not a standard benchmark.", 21, MUTED)
    text(draw, (70, 850), "Generated from saved AWS results; not an AWS console screenshot.", 20, MUTED)
    image.save(ASSETS / "scifact-comparison.png", optimize=True)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    results()
    live_checks()
    scifact_comparison()
    print("Wrote three generated results figures to", ASSETS)


if __name__ == "__main__":
    main()
