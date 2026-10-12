
from src.llm import generate_draft


def test_generate_draft_returns_text(monkeypatch):
    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": "fake-input"}

        def decode(self, output, skip_special_tokens=True):
            return "Market evidence summary."

    class FakeModel:
        def generate(self, **kwargs):
            return ["fake-output"]

    monkeypatch.setattr(
        "src.llm._load_model",
        lambda: (FakeTokenizer(), FakeModel()),
    )

    result = generate_draft(
        ticker="AAPL",
        research_question="Summarize market performance.",
        evidence={"market": {"latest_close": 336.64}},
    )

    assert isinstance(result, str)
    assert result == "Market evidence summary."