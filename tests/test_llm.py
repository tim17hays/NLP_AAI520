
import torch

from src.llm import generate_draft


def test_generate_draft_returns_text(monkeypatch):
    class FakeTensor:
        shape = (1, 10)

        def to(self, device):
            return self

        def __getitem__(self, index):
            return self

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ):
            assert len(messages) == 2
            return "Fake financial research prompt"

        def __call__(self, *args, **kwargs):
            return {
                "input_ids": FakeTensor(),
                "attention_mask": FakeTensor(),
            }

        def decode(self, output, skip_special_tokens=True):
            return "- Market evidence summary."

    class FakeModel:
        def generate(self, **kwargs):
            return FakeTensor()

    monkeypatch.setattr(
        "src.llm._load_model",
        lambda: (FakeTokenizer(), FakeModel()),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = generate_draft(
        ticker="AAPL",
        research_question="Summarize market performance.",
        evidence={"market": {"latest_close": 336.64}},
    )

    assert isinstance(result, str)
    assert result == "- Market evidence summary."