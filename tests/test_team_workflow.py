"""Test the actual shared planner/evaluator interface with labeled fixtures.

No model or data-provider calls. The scripted decisions establish plumbing,
not real-model quality or learning. Ashley's original notebook remains intact.
"""
import copy
import json
from pathlib import Path

from src import evaluation_memory as evaluation
from src.planning import ResearchPlanner
from src.workflow import run_two_run_experiment


def fixture_namespace():
    source = Path(__file__).resolve().parents[1] / "notebooks/evaluation_revision_memory.ipynb"
    notebook = json.loads(source.read_text())
    namespace = vars(evaluation).copy()
    # Fictional context and scripted model helpers, including their offline demo.
    for index in (17, 19):
        exec(compile("".join(notebook["cells"][index]["source"]),
                     f"ashley-fixture-cell-{index}", "exec"), namespace)
    return notebook, namespace


def test_ashley_acceptance_and_failure_checks_use_shared_module():
    notebook, namespace = fixture_namespace()
    # Reuse all ten acceptance/failure groups against the extracted module.
    exec(compile("".join(notebook["cells"][21]["source"]),
                 "ashley-offline-checks", "exec"), namespace)


def test_shared_planner_revision_and_persistent_memory_two_runs(monkeypatch, tmp_path):
    _, fixture = fixture_namespace()
    context = fixture["SAMPLE_CONTEXT"]
    draft_lessons = []

    class Result:
        def __init__(self, section):
            self.section = section

        def to_dict(self):
            return copy.deepcopy(context["evidence"][self.section])

    class Tools:
        def market_snapshot(self, ticker):
            return Result("market")

        def financial_snapshot(self, ticker):
            return Result("earnings")

    class News:
        def process(self, ticker):
            return copy.deepcopy(context["evidence"]["news"])

    def scripted_draft(**kwargs):
        draft_lessons.append(copy.deepcopy(kwargs["lessons"]))
        return fixture["fixture_draft"](context, kwargs["lessons"])

    monkeypatch.setattr("src.planning.generate_draft", scripted_draft)
    planner = ResearchPlanner(research_tools=Tools(), news_pipeline=News())
    memory_path = tmp_path / "research_lessons.json"
    arguments = dict(planner=planner, ticker="DEMO",
                     research_question=fixture["DEMO_QUESTION"],
                     call_model=fixture["fixture_model"], memory_path=memory_path)
    experiment = run_two_run_experiment(**arguments, output_dir=tmp_path / "outputs", execution_mode="offline_fixture")
    first, second = experiment["runs"]
    assert experiment["demonstration_complete"]
    assert experiment["new_run_one_lesson_used_in_run_two"]
    assert all(Path(p).exists() for p in experiment["run_files"])

    assert first["passed"] and first["revisions_completed"] == 1
    assert first["saved_lesson"] is not None
    assert second["passed"] and second["revisions_completed"] == 0
    assert draft_lessons[0] == []
    assert draft_lessons[1][0]["source_run_id"] == first["run_id"]
    assert second["plan"]["retrieved_lessons"] == second["lessons_used"]
    assert fixture["DEMO_LESSON"] in second["plan"]["steps"][0]
    assert second["evidence"] == context["evidence"]
    assert {step["specialist"] for step in second["tool_trace"]} == {"market", "earnings", "news"}


def test_no_lesson_does_not_report_complete_feedback_loop(tmp_path):
    _, fixture = fixture_namespace()

    class PassingPlanner(fixture["FixturePlanner"]):
        def execute(self, *args, **kwargs):
            result = super().execute(*args, **kwargs)
            result["draft_analysis"] = fixture["REFERENCE_ANALYSIS"]
            return result

    def passing_model(instructions, payload):
        return fixture["manual_evaluation"]([2, 2, 2, 2, 2])

    result = run_two_run_experiment(
        PassingPlanner(), "DEMO", fixture["DEMO_QUESTION"], passing_model,
        memory_path=tmp_path / "lessons.json", output_dir=tmp_path / "outputs",
        execution_mode="offline_fixture",
    )
    assert not result["demonstration_complete"]
    assert not result["new_run_one_lesson_used_in_run_two"]
    assert not result["revision_exercised"]
