"""Independent arithmetic checks for the documented evidence-brief schema.

These checks do not assign LLM scores or override model acceptance. They check
specific displayed calculations against raw inputs using Decimal arithmetic.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .evaluation_memory import check_citations
from .grounded_reporting import compile_evidence_brief


def verify_evidence_brief(context, analysis):
    """Check calculations independently, citations and constrained wording."""
    evidence = context["evidence"]
    earnings = evidence["earnings"]["data"]
    market = evidence["market"]["data"]
    try:
        current = Decimal(str(earnings["current_revenue"]))
        previous = Decimal(str(earnings["previous_revenue"]))
        income = Decimal(str(earnings["operating_income"]))
        first = Decimal(str(market["first_close"]))
        latest = Decimal(str(market["latest_close"]))
        if not all(
            value.is_finite()
            for value in (current, previous, income, first, latest)
        ):
            raise ValueError("Non-finite input")
        expected = {
            "revenue_growth": (current - previous) / previous * 100,
            "operating_margin": income / current * 100,
            "closing_price_change": (latest - first) / first * 100,
        }
    except (KeyError, InvalidOperation, ZeroDivisionError, ValueError) as exc:
        return {"passed": False, "input_error": str(exc)}
    patterns = {
        "revenue_growth": r"revenue growth:\s*([+-]?\d+(?:\.\d+)?)%",
        "operating_margin": r"operating margin(?: change)?:\s*([+-]?\d+(?:\.\d+)?)%",
        "closing_price_change": r"closing[- ]price (?:change|difference):\s*([+-]?\d+(?:\.\d+)?)%",
    }
    calculations = []
    for name, expression in patterns.items():
        matches = re.findall(expression, analysis, flags=re.IGNORECASE)
        values = [Decimal(value) for value in matches]
        valid = bool(values) and all(
            abs(value - expected[name]) < Decimal("0.0001") for value in values
        )
        calculations.append(
            {
                "metric": name,
                "expected_percent": str(expected[name]),
                "displayed_percent": [str(value) for value in values],
                "passed": valid,
            }
        )
    citations = check_citations(analysis, evidence)
    try:
        canonical = compile_evidence_brief(context)["analysis"]
        constrained_wording = analysis == canonical
    except ValueError:
        constrained_wording = False
    return {
        "calculations": calculations,
        "citation_check": citations,
        "matches_cited_evidence_compiler": constrained_wording,
        "passed": all(item["passed"] for item in calculations)
        and citations["passed"]
        and constrained_wording,
        "scope": "Three named percentage metrics, valid supplied references and exact schema-bound report wording; not general financial truth or model acceptance.",
    }


def verify_demonstration(report, baseline):
    """Separate observable policy benefits from the recorded model judgment."""
    runs = report["results"]
    finals = [
        verify_evidence_brief(baseline["context"], run["revised_analysis"])
        for run in runs
    ]
    before = verify_evidence_brief(baseline["context"], baseline["candidate"])
    after = verify_evidence_brief(
        baseline["context"], runs[1]["initial_draft"]
    )
    workflow = {
        key: value
        for key, value in report["checks"].items()
        if key
        not in ("run_1_passed", "run_2_passed", "run_2_needed_no_revision")
    }
    checks = {
        **workflow,
        "run_1_final_evidence_verified": finals[0]["passed"],
        "run_2_final_evidence_verified": finals[1]["passed"],
        "same_input_draft_improved_under_memory_policy": not before["passed"]
        and after["passed"],
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "final_evidence_checks": finals,
        "same_input_before": before,
        "same_input_after": after,
        "unchanged_model_acceptance": [run["passed"] for run in runs],
        "model_acceptance_note": "The model acceptance flags and scores remain unchanged. Evidence verification is a separate, explicitly scoped measurement.",
    }
