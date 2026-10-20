"""Test score provenance and exact-input caching without a model dependency."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.criterion_reviewer import CriterionReviewer
from src.evaluation_memory import CRITERIA


class Tensor:
    shape = (1, 10)

    def to(self, device):
        assert device == "cpu"
        return self


class Values:
    def __init__(self, scores):
        self.scores = scores

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.scores

    def argmax(self):
        return SimpleNamespace(
            item=lambda: self.scores.index(max(self.scores))
        )


class Logits:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        assert index == (0, -1, [100, 101, 102])
        return Values(self.values)


class Tokenizer:
    def encode(self, value, **kwargs):
        return [100 + int(value)]

    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def __call__(self, prompt, **kwargs):
        assert not kwargs["truncation"]
        return {"input_ids": Tensor()}


class Model:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=100)

    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        assert kwargs["logits_to_keep"] == 1 and kwargs["use_cache"] is False
        label = [0, 1, 2, 1, 2][self.calls]
        self.calls += 1
        values = [-3.0] * 3
        values[label] = 2.0
        return SimpleNamespace(logits=Logits(values))


def test_every_score_is_an_actual_model_selection_and_cache_is_exact(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    model = Model()
    reviewer = CriterionReviewer(Tokenizer(), model, "explicit-test-fixture")
    payload = {
        "task": "evaluate",
        "research_question": "Question",
        "evidence": {},
        "draft_analysis": "Candidate",
    }
    result = reviewer("Evaluate", payload)
    assert list(result["scores"]) == list(CRITERIA)
    assert list(result["scores"].values()) == [0, 1, 2, 1, 2]
    assert [int(call["response_text"]) for call in reviewer.calls] == [
        0,
        1,
        2,
        1,
        2,
    ]
    assert reviewer("Evaluate", payload) == result
    assert model.calls == 5 and len(reviewer.cache_events) == 1
    assert result["lesson"] == ""


def test_multitoken_labels_fail_instead_of_defaulting_scores():
    tokenizer = Tokenizer()
    tokenizer.encode = lambda *args, **kwargs: [1, 2]
    with pytest.raises(ValueError, match="single-token"):
        CriterionReviewer(tokenizer, Model(), "fixture")


def test_context_overflow_never_truncates_or_assigns_scores(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    model = Model()
    model.config = SimpleNamespace(max_position_embeddings=10)
    reviewer = CriterionReviewer(Tokenizer(), model, "fixture")
    with pytest.raises(ValueError, match="context"):
        reviewer(
            "Evaluate", {"task": "evaluate", "draft_analysis": "Candidate"}
        )
    assert reviewer.calls == [] and model.calls == 0
