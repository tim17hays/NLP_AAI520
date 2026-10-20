"""Check the final evidence measurement independently of model scores."""

import json
from pathlib import Path

from src.evidence_verification import verify_evidence_brief
from src.grounded_reporting import compile_evidence_brief


def context():
    path = (
        Path(__file__).parents[1]
        / "data/examples/grounded_baseline_input.json"
    )
    return json.loads(path.read_text())["context"]


def test_independent_decimal_calculations_and_citations_match_brief():
    source = context()
    text = compile_evidence_brief(source)["analysis"]
    result = verify_evidence_brief(source, text)
    assert result["passed"]
    assert [item["expected_percent"] for item in result["calculations"]] == [
        "20.0",
        "10.0",
        "10.0",
    ]


def test_tampered_number_and_appended_unsupported_claim_fail():
    source = context()
    text = compile_evidence_brief(source)["analysis"]
    assert not verify_evidence_brief(
        source, text.replace("Revenue growth: 20%", "Revenue growth: -20%")
    )["passed"]
    assert not verify_evidence_brief(
        source, text + "\nGuaranteed return of 25%."
    )["passed"]


def test_independent_check_tracks_changed_raw_inputs():
    source = context()
    source["evidence"]["earnings"]["data"]["current_revenue"] = 150
    text = compile_evidence_brief(source)["analysis"]
    result = verify_evidence_brief(source, text)
    assert result["passed"]
    assert result["calculations"][0]["expected_percent"] == "50.0"
