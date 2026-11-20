# AI Support Engine

Build an IAM-protected support search API using Amazon S3 Vectors, AWS Lambda, API Gateway, and a SageMaker Serverless Inference endpoint running `BAAI/bge-small-en-v1.5`. It compares keyword, vector, and native hybrid search and provides a separate exact-key lookup for a known document identifier. It returns matching documents, not generated answers.

The article and written benchmark will be published separately. This repository contains the implementation, synthetic fixtures, tests, saved results, and an optional offline GUI replay.

**Verified November 18, 2026 in `us-west-2`:** the model endpoint returned valid document and query embeddings, all 16 synthetic documents were indexed through S3 and Lambda, and signed keyword, vector, hybrid, and exact-key checks passed. The [model check](results/2026-11-18-revision-model-check.json), [API checks](results/2026-11-18-revision-smoke.json), and [infrastructure record](results/2026-11-18-revision-infrastructure.json) contain the observations. The earlier [32-query evaluation](results/2026-11-18-synthetic.json) repeated the November 17 aggregate rankings. This is retrieval, not an answer-generating application.

## Prerequisites and costs

- Python 3.10+, AWS CLI v2 for the documented commands, and an authenticated AWS profile. The Lambda functions use Python 3.13 and bundle the pinned SDK in [requirements.txt](requirements.txt).
- An AWS Region supporting **S3 Vectors enhanced indexes** and SageMaker Serverless Inference. The pinned serving image and complete workflow were tested in `us-west-2`; other Regions may not contain the same image digest.
- Permission to create and delete the resources in [template.yaml](template.yaml), manage tagged S3/S3 Vectors buckets and objects, inspect Service Quotas and the SageMaker public catalog/ECR, pass the SageMaker and Lambda roles, invoke the endpoint, and sign `execute-api:Invoke` requests.
- For the optional retry comparison, permission to read and update the lab search Lambda configuration and inspect the lab stack and endpoint; `--include-logs` additionally requires CloudWatch Logs read access.
- At least one available SageMaker serverless endpoint and one unit of concurrency, plus capacity for a **3,072 MB** endpoint. The quota preflight checks the first two limits, not available capacity or the per-endpoint memory cap. A successful deployment and model invocation are the real test. Your account's quotas can differ from ours.
- Internet access to download the pinned model files and Python dependencies. Use the synthetic fixtures or the optional licensed public SciFact and synthetic Bitext data; do not upload sensitive documents. The model weights are pinned to a revision and SHA-256-checked; the serving container image is pinned by digest.

This disposable lab incurs charges for SageMaker serverless inference, S3 Vectors, S3, Lambda, API Gateway, SQS, and CloudWatch. Review current regional pricing before deploying. It uses on-demand inference with no provisioned concurrency; idle on-demand endpoint compute is not billed, but stored artifacts and other resources may still incur charges. Delete the lab when finished.

Allow roughly 30–75 minutes for the main walkthrough through cleanup. A rounded, assumption-based [complete walkthrough estimate](results/2026-11-18-walkthrough-estimate.json) is about $0.101 before tax in `us-west-2`; it is not measured usage or a cost cap. Optional external benchmarks and repeated runs cost extra.

## Deploy and try it

Clone the repository and run the following from the project directory:

```bash
git clone https://github.com/ayogun/ai-support-engine.git
cd ai-support-engine

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export AWS_PROFILE=your-profile
aws sts get-caller-identity

python scripts/lab.py deploy --region us-west-2
PYTHONPATH=src:. python -m scripts.check_model --output results/my-model-check.json
python scripts/lab.py seed
python scripts/lab.py status
```

Deployment packages the code and pinned model, reviews a CloudFormation change set, creates the stack, then creates the enhanced text-searchable vector index. It can take several minutes. The model check must return two normalized 384-dimensional embeddings. `seed` uploads the 16 files from `fixtures/documents/` **one at a time** to the `documents/` prefix in the S3 bucket and waits for each source version to appear in the index, avoiding concurrent model invocations. `status` should report 16 ingested vectors.

The helper checks the active SDK's enhanced S3 Vectors support **before** creating resources. If it reports an old SDK, activate the intended virtual environment and reinstall `requirements.txt`. If any later deployment fails, retain `.lab-state.json` and use the ownership-checked cleanup; do not start a second deployment over it.

```bash
python scripts/lab.py query 'Find RETURN301' --mode keyword --limit 3
python scripts/lab.py query 'Find RETURN301' --mode vector --limit 3
python scripts/lab.py query 'Find RETURN301' --mode hybrid --limit 3
python scripts/lab.py query 'My order is still waiting to leave the warehouse.' --mode hybrid --limit 3
python scripts/lab.py query 'return301' --mode exact
```

The helper signs requests with the active AWS credentials. Keyword search and exact lookup skip model invocation; vector and hybrid search use the same endpoint as ingestion. Exact lookup is case-sensitive and accepts a **filename stem**, not arbitrary text: `return301` resolves `documents/return301.txt`, while `RETURN301` does not. IDs must be 1–128 ASCII letters, digits, `.`, `_`, or `-`, starting with a letter or digit; a missing indexed key returns an empty list. The endpoint allows one concurrent request and can cold-start, so this is deliberately not a load test or a production configuration.

Re-running `seed` skips unchanged, already-indexed object versions; it requeues an unchanged object if its vector is missing. It **refuses to overwrite a changed fixture**. Failed asynchronous events remain in the dead-letter queue until inspected, replayed separately, or deleted with the lab. The uploader is not a general document synchronization system.

## Test and evaluate

```bash
PYTHONPATH=src:. python -m scripts.smoke_test results/my-smoke.json
PYTHONPATH=src:. python -m scripts.inspect_ingestion results/my-ingestion.json
PYTHONPATH=src:. python -m scripts.evaluate --embedding sagemaker --output results/my-evaluation.json
```

The smoke test checks anonymous access, input validation, all three ranked search modes, and positive, missing, case-sensitive, and invalid exact IDs. Exit code `0` means all checks passed, `2` means infrastructure checks passed but a model-backed search failed, and `1` means another check failed. If the index is empty, the keyword and exact checks can temporarily use a synthetic non-model vector; it is removed and never used for semantic search. Check that 16 real fixture vectors are present before treating the test as end-to-end evidence.

The evaluator verifies the expected keys and waits for text indexing, embeds questions using the deployed endpoint, and runs all modes against the same 384-dimensional index. Its timings cover sequential direct S3 Vectors calls, **not** embeddings or API Gateway. `--embedding bge` is an optional local-model experiment that creates a separate index and does **not** test the SageMaker ingestion path; it requires `python -m pip install -r requirements-eval.txt`. Do not mix vectors from different embedding implementations or models in one index.

The saved SageMaker run used 16 hand-written documents and 32 hand-labeled questions. Keyword/vector/hybrid hit@1 were 27/32, 30/32, and 30/32 respectively; on this tiny synthetic set, hybrid corrected two identifier rankings but made two paraphrase rankings worse than vector-only search. These are illustrative results, not a production benchmark. Historical local-BGE and failed-Bedrock observations in `results/` are separate experiments, not evidence for the deployed model.

### Additional evaluations and bounded latency

Run these **one at a time**, after ingestion has finished; otherwise the shared one-concurrency endpoint can distort the measurements:

```bash
PYTHONPATH=src:. python -m scripts.evaluate_scifact --pool random --output results/my-scifact-random.json
PYTHONPATH=src:. python -m scripts.evaluate_scifact --pool lexical --output results/my-scifact-lexical.json
PYTHONPATH=src:. python -m scripts.evaluate_support --output results/my-support-identifiers.json
PYTHONPATH=src:. python -m scripts.evaluate_bitext --output results/my-bitext-support.json
PYTHONPATH=src:. python -m scripts.evaluate_techqa --output results/my-techqa-support.json
PYTHONPATH=src:. python -m scripts.probe_latency --output results/my-latency.json
PYTHONPATH=src:. python -m scripts.compare_retries --rounds-per-block 5 --include-logs --output results/my-retries.json
```

The identifier evaluation uses 16 synthetic support documents with punctuation, case, and lookalike IDs, plus 32 author-labeled questions and eight absent exact keys. Its separate index is populated directly. The [saved evaluation](results/2026-11-18-support-identifiers.json) reports search ranks and exact-key checks; the [API smoke test](results/2026-11-18-revision-smoke.json) verifies the exact route separately.

The Bitext evaluation downloads a pinned, SHA-256-verified copy of the [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), licensed under [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). The dataset is synthetic and the source CSV is cached in `.build/`, not included in the release. It selects 270 response documents and 54 held-out questions, balanced over 27 intent labels, and treats matching intent as a relevance **proxy**, not a human document-relevance judgment. A fresh run makes 324 model calls, plus direct vector queries and readiness checks. It is not a representative sample of real customer traffic.

The optional [TechQA evaluation](scripts/evaluate_techqa.py) downloads a pinned public dataset of historical IBM technical-support questions and linked documents. It constructs a 600-document pool containing all 100 selected linked documents. The [saved report](results/2026-11-18-techqa-support.json) and [paired comparison](results/2026-11-18-techqa-paired.json) include IDs and ranks, not source text. These original links are not new independent relevance judgments, and the constructed pool does not represent your organization's support traffic. A fresh run makes at least 700 model invocations and 900 vector queries and takes longer than the main walkthrough. The dataset publisher labels its derivative Apache-2.0; check the original terms and your own requirements before use. Source text is cached locally, not bundled in the release.

A fresh SciFact pool uses the actual endpoint to embed 300 abstracts and 50 claims, then makes 150 direct S3 Vectors queries. The archive is downloaded from the original authors and checked against a pinned SHA-256; it is cached under `.build/`, not bundled in the release. Each pool uses its own tagged index and is registered for cleanup before creation. On a partial retry, existing vectors are verified against the source text and model fingerprint before reuse. These evaluation indexes are populated **directly**, not through the S3-to-Lambda ingestion path.

The scripts select 50 evidence-bearing claims from SciFact's 300 development claims using a deterministic hash order. All 52 known evidence abstracts for that selection are included. The random pool fills the remaining slots by deterministic hash order; the lexical pool deliberately selects high-overlap background abstracts with a small local BM25 implementation, round-robin across claims. That challenge pool is biased by construction and is **not** a standard SciFact benchmark. Both pools force known evidence into a 300-document subset; unannotated documents can be relevant, no-evidence claims are excluded, and the model's training-data overlap is unknown. Results cannot be generalized to the complete corpus or support traffic.

The source is [SciFact](https://github.com/allenai/scifact), by Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020). Consult the original [data license](https://github.com/allenai/scifact/blob/master/LICENSE.md): the claims and abstracts have different license terms. The download URL contains `latest`; if it changes, the pinned checksum intentionally fails rather than silently changing the evaluation.

The latency probe makes 10 sequential requests per mode after separate warm-ups, followed by three bursts of three concurrent hybrid requests. It separates HTTP status from whether the expected document was retrieved and reports HTTP-response and success-only percentiles. Older [saved runs](results/2026-11-18-latency.json) used transport-only success. The bounded retry comparison changes only the search function's SDK attempt count in ABBA order, records the same small workload, and restores the original environment; forcibly terminating the process can prevent restoration. See the [pooled comparison](results/2026-11-18-retry-pooled.json). Neither tool is a load test or an SLA.

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
cfn-lint template.yaml
cfn-guard validate --rules checks/lab.guard --data template.yaml --show-summary all
```

`cfn-lint` and `cfn-guard` are separately installed developer tools. To regenerate the three result illustrations from saved results, install [requirements-eval.txt](requirements-eval.txt) and run `python scripts/render_figures.py`. The AWS architecture illustration has a separate [editable draw.io source](assets/architecture.drawio) and [diagram guide](assets/architecture.md). Sanitized console and terminal captures are in [assets/](assets/).

## Optional offline GUI

The [local replay](demo/) makes the support-search example easier to explore. It uses the saved synthetic fixture and recorded direct-index results, not the deployed API. The exact lookup requires a case-sensitive, already-known document ID; it does not extract an ID from a question. Only recorded questions and IDs are available. The screenshots are illustrations, not evidence of live requests.

Start the preview with Python 3 and open `http://127.0.0.1:8765/`:

```bash
python3 demo/server.py
```

The server binds to localhost, serves only its allowlisted files and the saved examples, and makes no AWS requests. It does not need or read AWS credentials. Press `Ctrl+C` to stop it. To use another local port, add `--port 8766`.

## Cleanup

```bash
python scripts/lab.py cleanup
```

This permanently removes the lab's synthetic documents, evaluation indexes, endpoint, API, logs, queue, stack, and deployment artifacts. Cleanup verifies the active account and ownership tags on the stack, buckets, and registered indexes and refuses unexpected object prefixes. If cleanup fails, **keep `.lab-state.json`**, inspect the error, and retry; it is needed for safe recovery. Check for residual resources and charges in your account.

The [cleanup record](results/2026-11-18-revision-cleanup.json) verifies that the follow-up lab's stack, endpoint, model, buckets, queue, and API are absent; it is not a billing statement.

## Build a project archive

Build and verify the local release candidate after completing the evidence files:

```bash
python scripts/create_release.py
python scripts/verify_release.py dist/serverless-hybrid-search-v1.3.0.zip --run-tests
```

The archive uses an explicit allowlist and SHA-256 manifest. Verification checks the manifest and relative links, then runs tests in a clean temporary extraction without AWS credentials. It excludes the article and written benchmark, `.env`, `.lab-state.json`, `.build/`, virtual environments, and unlisted personal results. Inspect the archive before sharing; the access-key scan is not a comprehensive secret scanner. The original source code and infrastructure templates use the [MIT license](LICENSE). Screenshots, illustrations, and results are not separately licensed by that code grant; the external model and datasets retain their own licenses and terms.

## Scope

Single tenant; synthetic English application fixtures; immutable fixture keys. The optional benchmarks send public SciFact, synthetic Bitext, and public historical TechQA data to the endpoint and store their text in separate vector indexes. No generated answers, document chunking, safe update/delete reconciliation, tenant authorization, production-grade replay, or load-tested rate controls are implemented. Do not use this lab with sensitive data without addressing those gaps.
