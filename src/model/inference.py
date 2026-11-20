import json


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TEXT_BYTES = 8000


def prepare_text(payload):
    if not isinstance(payload, dict):
        raise ValueError("The request must be a JSON object.")
    text = payload.get("text")
    input_type = payload.get("input_type")
    if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("Text must contain 1–8000 UTF-8 bytes.")
    if input_type not in {"document", "query"}:
        raise ValueError("input_type must be document or query.")
    return QUERY_PREFIX + text if input_type == "query" else text


def model_fn(model_dir):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
    )
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "torch": torch}


def transform_fn(model, input_data, content_type, accept):
    if content_type != "application/json" or accept != "application/json":
        raise ValueError("Only application/json is supported.")
    text = prepare_text(json.loads(input_data))
    tokens = model["tokenizer"](text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with model["torch"].inference_mode():
        output = model["model"](**tokens).last_hidden_state[:, 0]
        vector = model["torch"].nn.functional.normalize(output, p=2, dim=1)[0].tolist()
    return json.dumps({"embedding": vector}, allow_nan=False)
