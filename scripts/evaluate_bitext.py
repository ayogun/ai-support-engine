#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from scripts import evaluate, evaluate_scifact, evaluate_support, lab
from search_lab import core


DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"
FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILENAME}"
DATASET_SHA256 = "6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434"
DATASET_LIMIT = 25 * 1024 * 1024
SEED = "20261118"
INDEX_NAME = "bitext-support"
DOCUMENTS_PER_INTENT = 10
QUERIES_PER_INTENT = 2
TOP_K = 3


def dataset_path(path):
    path = Path(path)
    if path.exists():
        if lab.sha256_file(path) != DATASET_SHA256:
            raise RuntimeError("The cached Bitext dataset checksum does not match the pinned revision.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=120) as response, temporary.open("wb") as output:
            if urllib.parse.urlparse(response.geturl()).scheme != "https":
                raise RuntimeError("The Bitext dataset download redirected to an insecure URL.")
            size = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > DATASET_LIMIT:
                    raise RuntimeError("The Bitext dataset exceeds the download size limit.")
                output.write(block)
        if lab.sha256_file(temporary) != DATASET_SHA256:
            raise RuntimeError("The downloaded Bitext dataset does not match the pinned revision.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_rows(path):
    rows = []
    with dataset_path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"instruction", "intent", "response"}.issubset(reader.fieldnames or []):
            raise RuntimeError("The Bitext dataset is missing required columns.")
        for position, row in enumerate(reader):
            rows.append({"row": position, "question": row["instruction"],
                         "intent": row["intent"], "text": row["response"]})
    return rows


def order_key(namespace, row):
    return hashlib.sha256(f"{SEED}:{namespace}:{row['row']}".encode()).digest(), row["row"]


def select_dataset(rows, documents_per_intent=DOCUMENTS_PER_INTENT, queries_per_intent=QUERIES_PER_INTENT):
    if documents_per_intent < 1 or queries_per_intent < 1:
        raise ValueError("The number of documents and queries per intent must be positive.")
    grouped = defaultdict(list)
    seen_rows = set()
    for row in rows:
        if row["row"] in seen_rows:
            raise ValueError("Dataset row IDs must be unique.")
        seen_rows.add(row["row"])
        if (row["intent"].strip() and row["question"].strip() and row["text"].strip()
                and len(row["question"].encode("utf-8")) <= core.MAX_QUERY_BYTES
                and len(row["text"].encode("utf-8")) <= core.MAX_DOCUMENT_BYTES - 40):
            grouped[row["intent"]].append(row)
    if not grouped:
        raise ValueError("No eligible Bitext rows were found.")
    selected_documents = []
    used_responses = set()
    document_questions = set()
    selected_rows = set()
    for intent in sorted(grouped):
        selected = 0
        for row in sorted(grouped[intent], key=lambda item: order_key("document", item)):
            response_key = row["text"].strip().casefold()
            if response_key in used_responses:
                continue
            selected_documents.append(row)
            used_responses.add(response_key)
            document_questions.add(row["question"].strip().casefold())
            selected_rows.add(row["row"])
            selected += 1
            if selected == documents_per_intent:
                break
        if selected != documents_per_intent:
            raise ValueError(f"Not enough distinct responses for intent {intent}.")
    document_ids = defaultdict(list)
    documents = []
    readiness = {}
    for position, row in enumerate(selected_documents):
        identifier = f"bitext-{row['row']}"
        marker = f"zzbitextready{position:04d}"
        documents.append((f"bitext/{identifier}.txt", f"{row['text']}\nReadiness marker: {marker}."))
        document_ids[row["intent"]].append(identifier)
        readiness[identifier] = marker
    queries = []
    used_questions = set()
    for intent in sorted(grouped):
        selected = 0
        for row in sorted(grouped[intent], key=lambda item: order_key("query", item)):
            question_key = row["question"].strip().casefold()
            if row["row"] in selected_rows or question_key in document_questions or question_key in used_questions:
                continue
            queries.append({"id": f"bitext-query-{row['row']}", "question": row["question"],
                            "intent": intent, "relevant": document_ids[intent]})
            used_questions.add(question_key)
            selected += 1
            if selected == queries_per_intent:
                break
        if selected != queries_per_intent:
            raise ValueError(f"Not enough held-out questions for intent {intent}.")
    return documents, queries, readiness, len(grouped)


def run(arguments):
    rows = load_rows(arguments.dataset)
    documents, queries, readiness, intent_count = select_dataset(rows)
    session = lab.session_for_state()
    state = lab.load_state(session)
    vectors = session.client("s3vectors")
    runtime = session.client("sagemaker-runtime", config=Config(read_timeout=90, retries={"total_max_attempts": 1}))
    evaluate_scifact.prepare_index(vectors, state, documents, runtime, INDEX_NAME, "Bitext support")
    evaluate_support.wait_for_text_search(vectors, state, readiness, timeout=300, index_name=INDEX_NAME)
    observations = []
    for query in queries:
        embedding = core.embed_text(query["question"], state["outputs"]["EmbeddingEndpointName"],
                                    input_type="query", runtime=runtime)
        for mode in evaluate.MODES:
            request = core.query_request(state["vector_bucket"], INDEX_NAME, query["question"], mode, TOP_K, embedding)
            started = time.perf_counter()
            response = vectors.query_vectors(**request)
            elapsed = (time.perf_counter() - started) * 1000
            returned = response.get("vectors", [])
            identifiers = [item.get("metadata", {}).get("document_id") for item in returned]
            rank = next((position for position, identifier in enumerate(identifiers, 1)
                         if identifier in query["relevant"]), None)
            observations.append({"query_id": query["id"], "intent": query["intent"], "mode": mode,
                                 "relevant": query["relevant"], "returned_ids": identifiers,
                                 "first_relevant_rank": rank, "query_ms": round(elapsed, 2)})
        print("Evaluated support intent query", query["id"], flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "region": state["region"],
        "dataset": DATASET, "dataset_revision": REVISION, "dataset_url": DATASET_URL,
        "dataset_sha256": DATASET_SHA256, "dataset_license": "CDLA-Sharing-1.0",
        "source_row_count": len(rows), "selected_data_sha256": evaluate_scifact.corpus_digest(documents, queries),
        "selection": "Fixed SHA-256 ordering of zero-based CSV row numbers; per intent, select 10 distinct response texts and 2 held-out instruction rows. Query text is not an exact duplicate of a selected document row's instruction. A unique readiness marker is appended to each indexed response.",
        "seed": SEED, "intent_count": intent_count, "documents_per_intent": DOCUMENTS_PER_INTENT,
        "queries_per_intent": QUERIES_PER_INTENT, "document_count": len(documents), "query_count": len(queries),
        "top_k": TOP_K, "model": core.MODEL_ID, "model_revision": lab.MODEL_REVISION,
        "model_fingerprint": evaluate_scifact.MODEL_FINGERPRINT,
        "label_definition": "An indexed response is counted as a hit if its source row has the same Bitext intent as the held-out question; this is a proxy, not a document-relevance judgment.",
        "limitations": "Synthetic question/answer data, not a real support knowledge base; balanced deterministic sample is not actual traffic. Same-intent responses can be unsuitable or generic, and other intents can be relevant. Row holdout and exact text deduplication do not eliminate paraphrase, template, or model-training overlap.",
        "timing_scope": "Sequential direct S3 Vectors calls; excludes model inference and API Gateway.",
        "summary": evaluate_scifact.summarize(observations), "observations": observations,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print("Wrote", output)


def main():
    parser = argparse.ArgumentParser(description="Evaluate intent matching on a pinned, independently authored synthetic support dataset.")
    parser.add_argument("--dataset", default=".build/bitext-customer-support-27k.csv")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
        return 0
    except (BotoCoreError, ClientError, RuntimeError, ValueError, OSError, csv.Error) as error:
        print(f"Bitext evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
