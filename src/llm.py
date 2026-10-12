
"""Local LLM draft generation for financial research."""

import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-small"

_tokenizer = None
_model = None


def _load_model():
    """Load the local model once and reuse it."""
    global _tokenizer, _model

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.eval()

    return _tokenizer, _model



def generate_draft(
    ticker: str,
    research_question: str,
    evidence: dict,
    lessons: list | None = None,
) -> str:
    """Generate a concise research draft from selected evidence."""
    tokenizer, model = _load_model()

    # Keep the context compact so the model can process it.
    compact_evidence = {}

    for section, details in evidence.items():
        if isinstance(details, dict):
            compact_evidence[section] = {
                key: value
                for key, value in details.items()
                if key.lower() in {
                    "data", "summary", "source", "sources",
                    "error", "status", "sentiment", "latest_close",
                    "period_return_pct", "observations", "revenue",
                    "earnings", "news", "date"
                }
            }
        else:
            compact_evidence[section] = str(details)[:500]

    prompt = (
        f"Task: {research_question}\n"
        f"Stock: {ticker}\n"
        f"Evidence: {json.dumps(compact_evidence, default=str)[:1800]}\n"
        f"Lessons: {json.dumps(lessons or [])[:300]}\n"
        "Write a short financial research summary in 3 bullet points. "
        "Use only facts clearly present in the evidence. Never invent figures. "
        "If evidence is insufficient, state that. No buy or sell recommendation."
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)