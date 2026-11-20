# Serverless search with native S3 Vectors

The [editable diagram](architecture.drawio) describes the search lab implemented in [the template](../template.yaml) and [the deployment helper](../scripts/lab.py). It is an architecture illustration, not an AWS console screenshot.

## Ingest a document

1. A client uploads a `.txt` file under `documents/` to the versioned S3 document bucket.
2. An S3 object-created event invokes the ingestion Lambda asynchronously. The function reads the indicated object version and invokes the SageMaker Serverless endpoint to embed its text.
3. The Lambda writes the embedding and searchable text to the enhanced S3 Vectors index. After an asynchronous ingestion failure, Lambda can place the event in the SQS dead-letter queue.

## Search the index

1. A client sends an AWS Signature Version 4 signed request to the IAM-authorized API Gateway HTTP API. API Gateway invokes the search Lambda.
2. For vector and hybrid modes, the Lambda uses the same SageMaker endpoint to embed the question. Keyword and exact modes do not invoke the model.
3. Keyword, vector, and hybrid searches use `QueryVectors`. Hybrid sends both the text and the embedding for native S3 Vectors ranking. Exact lookup uses `GetVectors` with a key derived from the supplied case-sensitive document ID.
4. The Lambda returns matching document metadata through the API. It does not generate an answer.

The SageMaker endpoint and S3 Vectors index each appear twice in the diagram to keep the two paths legible. Each pair represents the same shared resource. Dashed lines indicate conditional or asynchronous interactions; the red dashed line indicates a failed ingestion event.

The lab uses small text files and a one-concurrency serverless endpoint. The exact path accepts a known filename stem and does not extract identifiers from free text. Safe update and delete reconciliation, a user interface, and production-grade access controls are outside its scope.
