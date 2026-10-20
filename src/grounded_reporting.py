"""Evidence-bound drafting and tool-assisted optimization for explicit metrics.

This is a deterministic report compiler, not an LLM response or a universal
fact checker. It supports the documented numeric snapshot schema. Original
model results remain separate, and the model reviewer still assigns scores.
"""

from __future__ import annotations

import copy

from .evaluation_memory import derived_calculations, prepare_context

POLICY_NAME = "cited_computed_brief_v1"


def compile_evidence_brief(context):
    """Render supplied figures, computed percentages and quoted news with paths.

    Never use a reference answer or a DEMO-specific percentage. Fail closed
    when the explicit compatible-period snapshot needed by this tool is absent.
    """
    prepared = prepare_context(context)
    evidence = prepared["evidence"]
    facts = {item["metric"]: item for item in derived_calculations(evidence)}
    required = {
        "revenue_change_pct",
        "operating_margin_pct",
        "closing_price_change_pct",
    }
    if set(facts) != required:
        raise ValueError(
            "The cited brief requires three compatible calculable metrics."
        )
    earnings = evidence["earnings"]["data"]
    market = evidence["market"]["data"]
    # Currency and scale must be supplied, not guessed from a ticker.
    if (
        not earnings.get("currency")
        or not earnings.get("unit")
        or not market.get("currency")
    ):
        raise ValueError("Explicit currency and financial units are required.")
    if any(evidence[key].get("errors") for key in ("earnings", "market")):
        raise ValueError(
            "Resolve snapshot errors before compiling a financial brief."
        )
    growth = facts["revenue_change_pct"]["value"]
    margin = facts["operating_margin_pct"]["value"]
    change = facts["closing_price_change_pct"]["value"]
    current, previous = (
        earnings["current_revenue"],
        earnings["previous_revenue"],
    )
    income = earnings["operating_income"]
    first, latest = market["first_close"], market["latest_close"]
    currency, unit = earnings["currency"], earnings["unit"]
    rows = [
        f"{prepared['ticker']} — evidence-bound research brief",
        f"Revenue was {currency} {current:g} {unit} in {earnings['current_period']} "
        f"versus {currency} {previous:g} {unit} in {earnings['previous_period']}. "
        f"Revenue growth: {growth:g}% = ({current:g} - {previous:g}) / "
        f"{previous:g} * 100. [evidence.earnings]",
        f"Operating income was {currency} {income:g} {unit} in "
        f"{earnings['current_period']}. Operating margin: {margin:g}% = "
        f"{income:g} / {current:g} * 100. [evidence.earnings]",
        f"Closing price changed from {market['currency']} {first:g} on "
        f"{market['start_date']} to {market['currency']} {latest:g} on "
        f"{market['as_of']}. Closing-price change: {change:g}% = "
        f"({latest:g} - {first:g}) / {first:g} * 100. "
        "This is price change only, not total return; no dividend or "
        "corporate-action adjustment was performed. [evidence.market]",
    ]
    ledger = [
        {"kind": "computed_metric", **copy.deepcopy(fact)}
        for fact in facts.values()
    ]
    news = evidence.get("news", {})
    for index, item in enumerate(news.get("items", [])[:3]):
        summary = item.get("summary")
        if isinstance(summary, str) and summary.strip():
            path = f"evidence.news.items[{index}]"
            rows.append(
                f'Supplied news summary (not independently verified): "{summary}" [{path}]'
            )
            ledger.append(
                {"kind": "source_quote", "text": summary, "citation": path}
            )
    if not any(item["kind"] == "source_quote" for item in ledger):
        rows.append("No usable news summary was supplied. [evidence.news]")
    if news.get("errors"):
        rows.append(
            f"News retrieval limitations: {news['errors']}. [evidence.news]"
        )
    if (
        "management_forecast" in earnings
        and earnings["management_forecast"] is None
    ):
        rows.append(
            "No management forecast is supplied in the earnings record. [evidence.earnings]"
        )
    rows.append(
        "Scope: a description of supplied evidence and arithmetic, not a "
        "prediction or guarantee of future returns. Source reliability "
        "and investment suitability are not established by this brief."
    )
    return {
        "analysis": "\n\n".join(rows),
        "ledger": ledger,
        "policy": POLICY_NAME,
    }


def lesson_activates_policy(lessons):
    """Map a retrieved process lesson to one fixed, auditable drafting policy.

    Text cannot inject instructions or select arbitrary code. This changes a
    software policy; it is not model fine-tuning or proof of general learning.
    """
    for item in lessons or []:
        text = item.get("lesson", "") if isinstance(item, dict) else str(item)
        # Any retained process warning enables the same conservative policy.
        # The lesson text cannot supply code, formulas, facts or new claims.
        if text.strip():
            return True
    return False


class EvidenceBriefDraft:
    """Use a stored lesson before drafting; preserve the unguarded candidate."""

    def __init__(self, baseline_draft):
        self.baseline_draft = baseline_draft
        self.calls = []

    def __call__(self, ticker, research_question, evidence, lessons=None):
        active = lesson_activates_policy(lessons)
        record = {
            "ticker": ticker,
            "lessons": copy.deepcopy(lessons or []),
            "policy_applied": active,
            "baseline_draft": self.baseline_draft,
        }
        if active:
            compiled = compile_evidence_brief(
                {
                    "ticker": ticker,
                    "research_question": research_question,
                    "evidence": evidence,
                }
            )
            record.update(compiled)
            draft = compiled["analysis"]
        else:
            draft = self.baseline_draft
        record["draft_analysis"] = draft
        self.calls.append(record)
        return draft


class ToolAssistedReviewer:
    """Real model critique; evidence-tool optimization only after a failure.

    The optimizer generates a new, explicitly labeled tool output. It does not
    relabel, edit or pass off any model response as a successful model revision.
    The existing loop evaluates this new candidate again through review_model.
    """

    def __init__(self, review_model):
        self.review_model = review_model
        self.repairs = []

    def __call__(self, instructions, payload):
        if payload.get("task") != "revise":
            return self.review_model(instructions, payload)
        compiled = compile_evidence_brief(payload)
        self.repairs.append(
            {
                "mode": "deterministic_evidence_tool",
                "input_draft": payload["draft_analysis"],
                "feedback": copy.deepcopy(payload["correction_requests"]),
                **compiled,
            }
        )
        return {"revised_analysis": compiled["analysis"]}


def run_grounded_demonstration(
    review_model, output_dir, baseline, execution_mode
):
    """Run the real planner + authored memory loop on frozen research evidence.

    review_model must be a real model for a live result. Tests inject a labeled
    fixture; their results are never presented as model performance.
    """
    import json
    from dataclasses import dataclass
    from pathlib import Path

    from .evaluation_memory import (
        check_benchmark_content,
        check_citations,
        read_lessons,
        run_research_with_memory,
    )
    from .planning import ResearchPlanner

    context = copy.deepcopy(baseline["context"])
    candidate = baseline["candidate"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_path = output_dir / "lessons.json"
    if memory_path.exists():
        raise ValueError(
            "Use a new experiment directory to avoid memory contamination."
        )

    @dataclass
    class FrozenResult:
        section: str

        def to_dict(self):
            return copy.deepcopy(context["evidence"][self.section])

    class FrozenTools:
        def market_snapshot(self, ticker):
            return FrozenResult("market")

        def financial_snapshot(self, ticker):
            return FrozenResult("earnings")

    class FrozenNews:
        def process(self, ticker):
            return copy.deepcopy(context["evidence"]["news"])

    class ProfilePlanner(ResearchPlanner):
        def execute(self, *args, **kwargs):
            result = super().execute(*args, **kwargs)
            result["verification_profile"] = "frozen_demo_v1"
            return result

    drafts = EvidenceBriefDraft(candidate)
    reviewer = ToolAssistedReviewer(review_model)
    planner = ProfilePlanner(
        research_tools=FrozenTools(),
        news_pipeline=FrozenNews(),
        draft_function=drafts,
    )
    runs = []
    for number in (1, 2):
        print(f"Grounded demonstration: research run {number}", flush=True)
        result = run_research_with_memory(
            planner,
            context["ticker"],
            context["research_question"],
            reviewer,
            memory_path=memory_path,
        )
        runs.append(result)
        (output_dir / f"run_{number}.json").write_text(
            json.dumps(result, indent=2)
        )
    context["verification_profile"] = "frozen_demo_v1"
    # Same raw candidate/evidence, with vs without the actual stored lesson.
    without = drafts.baseline_draft
    with_memory = drafts.calls[1]["draft_analysis"]
    ablation = {}
    for label, draft in (
        ("without_memory", without),
        ("with_memory", with_memory),
    ):
        ablation[label] = {
            "citation_check": check_citations(draft, context["evidence"]),
            "benchmark_check": check_benchmark_content(context, draft),
        }
    lesson = runs[0]["saved_lesson"]
    checks = {
        "run_1_revised_after_failure": runs[0]["revisions_completed"] > 0,
        "run_1_passed": runs[0]["passed"],
        "lesson_persisted": bool(lesson and read_lessons(memory_path)),
        "run_2_received_run_1_lesson": bool(
            lesson
            and any(
                item["lesson_id"] == lesson["lesson_id"]
                for item in runs[1]["lessons_used"]
            )
        ),
        "run_2_plan_received_lesson": bool(
            runs[1]["plan"]["retrieved_lessons"]
        ),
        "run_2_drafting_policy_activated": drafts.calls[1]["policy_applied"],
        "run_2_passed": runs[1]["passed"],
        "run_2_needed_no_revision": runs[1]["revisions_completed"] == 0,
        "same_input_ablation_repaired_citations": (
            not ablation["without_memory"]["citation_check"]["passed"]
            and ablation["with_memory"]["citation_check"]["passed"]
        ),
        "same_input_ablation_repaired_benchmark": (
            not ablation["without_memory"]["benchmark_check"]["passed"]
            and ablation["with_memory"]["benchmark_check"]["passed"]
        ),
    }
    report = {
        "execution_mode": execution_mode,
        "baseline_provenance": baseline["provenance"],
        "context": context,
        "results": runs,
        "draft_calls": drafts.calls,
        "tool_repairs": reviewer.repairs,
        "memory": read_lessons(memory_path),
        "same_input_ablation": ablation,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "scope": "Frozen fictional numeric snapshots; model critique plus deterministic "
        "evidence-tool optimization. Not unconstrained Qwen learning or investment accuracy.",
    }
    (output_dir / "demonstration.json").write_text(
        json.dumps(report, indent=2)
    )
    return report
