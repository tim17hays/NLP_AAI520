"""Independent arithmetic, provenance and memory-policy regression checks."""

import copy
import json
from pathlib import Path

import pytest

from src import evaluation_memory as em
from src.grounded_reporting import (
    compile_evidence_brief,
    run_grounded_demonstration,
)


def baseline():
    return json.loads(
        (
            Path(__file__).parents[1]
            / "data/examples/grounded_baseline_input.json"
        ).read_text()
    )


def test_cited_brief_repairs_observed_claims_without_reference_answer():
    context = baseline()["context"]
    before = copy.deepcopy(context)
    result = compile_evidence_brief(context)
    text = result["analysis"]
    assert "Revenue growth: 20%" in text and "Operating margin: 10%" in text
    assert "Closing-price change: 10%" in text
    assert "not total return" in text
    assert context == before
    assert em.check_citations(text, context["evidence"])["passed"]
    assert em.check_benchmark_content(
        {**context, "verification_profile": "frozen_demo_v1"}, text
    )["passed"]
    assert len(result["ledger"]) == 4
    assert (
        result["ledger"][-1]["text"]
        == context["evidence"]["news"]["items"][0]["summary"]
    )


def test_cited_brief_calculates_new_inputs_instead_of_copying_benchmark():
    context = baseline()["context"]
    data = context["evidence"]["earnings"]["data"]
    data.update(current_revenue=150, operating_income=30)
    context["evidence"]["market"]["data"]["latest_close"] = 48
    text = compile_evidence_brief(context)["analysis"]
    assert "Revenue growth: 50%" in text
    assert "Operating margin: 20%" in text
    assert "Closing-price change: 20%" in text


@pytest.mark.parametrize(
    "change", ["missing", "error", "units", "bool", "zero"]
)
def test_incomplete_or_invalid_inputs_fail_closed(change):
    context = baseline()["context"]
    data = context["evidence"]["earnings"]["data"]
    if change == "missing":
        del data["current_period"]
    if change == "error":
        context["evidence"]["earnings"]["errors"] = ["Unavailable"]
    if change == "units":
        data["unit"] = None
    if change == "bool":
        data["current_revenue"] = True
    if change == "zero":
        data["previous_revenue"] = 0
    with pytest.raises(ValueError):
        compile_evidence_brief(context)


def test_memory_changes_drafting_and_reduces_repairs_in_labeled_fixture(
    tmp_path,
):
    def fixture_review(instructions, payload):
        assert payload["task"] == "evaluate"
        # This scripted judgment only tests integration. Real model results are separate.
        return {
            "scores": dict.fromkeys(em.CRITERIA, 2),
            "feedback": [],
            "critical_issues": [],
            "lesson": "",
        }

    report = run_grounded_demonstration(
        fixture_review, tmp_path / "fixture", baseline(), "offline_fixture"
    )
    assert report["all_checks_passed"]
    assert report["results"][0]["revisions_completed"] == 1
    assert report["results"][1]["revisions_completed"] == 0
    assert len(report["tool_repairs"]) == 1
    assert (
        report["memory"][0]["source_run_id"] == report["results"][0]["run_id"]
    )
    assert report["draft_calls"][0]["policy_applied"] is False
    assert report["draft_calls"][1]["policy_applied"] is True
    with pytest.raises(ValueError):
        run_grounded_demonstration(
            fixture_review, tmp_path / "fixture", baseline(), "offline_fixture"
        )
