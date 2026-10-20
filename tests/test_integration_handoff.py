"""Offline integration checks; no paid model or provider calls."""
from src.data_tools import ToolResult
from src.news_pipeline import NewsPipeline
from src.planning import ResearchPlanner


class Tools:
    def financial_snapshot(self, ticker):
        return ToolResult("financial_snapshot", ticker, {"totalRevenue": 100})


def test_lesson_reaches_plan_and_draft_with_errors_retained(monkeypatch):
    calls = []
    monkeypatch.setattr("src.planning.generate_draft", lambda **kw: calls.append(kw) or "Draft")
    planner = ResearchPlanner(research_tools=Tools())
    first = planner.execute("AAPL", "financial metrics")
    lesson = "Do not infer reporting periods."
    second = planner.execute("AAPL", "financial metrics", lessons=[lesson])
    assert first["retrieved_lessons"] == []
    assert second["plan"]["retrieved_lessons"] == [lesson]
    assert lesson in second["plan"]["steps"][0]
    assert calls[1]["lessons"] == [lesson]
    assert calls[1]["evidence"] == second["evidence"]
    assert second["tool_trace"][0]["tool"] == "financial_snapshot"


def test_news_trace_links_facts_to_retained_source():
    class Classifier:
        def classify(self, text):
            return "neutral", 0.8
    raw = [{"title": "Apple revenue report", "summary": "Revenue was 100.",
            "link": "https://example.com/a?tracking=1", "publisher": "Fixture"}]
    result = NewsPipeline(classifier=Classifier()).process("AAPL", raw + raw)
    trace = result["trace"]
    assert trace["ingest"]["input_count"] == 2
    assert trace["preprocess"]["retained_count"] == 1
    assert trace["extract"][0]["url"] == result["items"][0]["source"]["url"]
    assert all(fact in trace["preprocess"]["records"][0]["text"]
               for fact in trace["extract"][0]["facts"])
    assert trace["summarize"] == result["summary"]


def test_prompt_retains_late_source_errors_and_lessons(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from src.llm import generate_draft
    captured = []
    class Tensor:
        shape = (1, 2)
        def to(self, device):
            assert device == "cuda"
            return self
        def __getitem__(self, index):
            return self
    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            captured.extend(messages)
            return "prompt"
        def __call__(self, *args, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": Tensor()}
        def decode(self, *args, **kwargs):
            return "Draft with sources"
    class Model:
        device = "cuda"

        def generate(self, **kwargs):
            return Tensor()
    monkeypatch.setattr("src.llm._load_model", lambda: (Tokenizer(), Model()))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    evidence = {"market": {"data": {"description": "x" * 2000}},
                "news": {"items": [{"source": {"url": "https://example.com/last"}}],
                         "errors": ["missing article bodies"]}}
    generate_draft("AAPL", "market and news", evidence, ["Do not infer periods"])
    prompt = captured[1]["content"]
    assert "https://example.com/last" in prompt
    assert "missing article bodies" in prompt
    assert "Do not infer periods" in prompt
