
"""Qwen-based draft generation for financial research."""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_model = None


def _load_model():
    """Load Qwen once, using the Colab GPU."""
    global _tokenizer, _model

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for Qwen inference. "
            "Run this project in Google Colab with a GPU runtime."
        )

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
        ).to("cuda")
        _model.eval()

    return _tokenizer, _model


def generate_draft(
    ticker: str,
    research_question: str,
    evidence: dict,
    lessons: list | None = None,
) -> str:
    """Generate a concise financial research draft from evidence."""
    tokenizer, model = _load_model()

    compact_evidence = {}

    allowed_keys = {
        "data", "summary", "source", "sources", "error", "status",
        "sentiment", "latest_close", "period_return_pct",
        "observations", "revenue", "earnings", "news", "date",
        "gross_profit", "net_income", "trailing_pe",
        "items_analyzed", "sentiment_counts", "limitation",
    }

    for section, details in evidence.items():
        if isinstance(details, dict):
            compact_evidence[section] = {
                key: value
                for key, value in details.items()
                if key.lower() in allowed_keys
            }
        else:
            compact_evidence[section] = str(details)[:500]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful financial research assistant. "
                "Use only facts explicitly supported by the supplied evidence. "
                "Do not invent figures, news events, trends, or explanations. "
                "Distinguish reported facts from interpretations. "
                "If evidence is insufficient, say so. "
                "Do not give buy or sell recommendations. "
                "Return exactly three concise bullet points."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {research_question}\n"
                f"Stock ticker: {ticker}\n"
                f"Evidence: {json.dumps(compact_evidence, default=str)[:1800]}\n"
                f"Lessons from previous evaluations: "
                f"{json.dumps(lessons or [], default=str)[:300]}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens, not the input prompt.
    input_length = inputs["input_ids"].shape[-1]
    new_tokens = output[0][input_length:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()