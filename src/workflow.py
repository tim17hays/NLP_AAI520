"""Shared team entry point; real results stay distinct from test fixtures.

Uses Tim's retrieval, Roopleen's planner/draft and Ashley's evaluator/memory.
No inference is performed when this module is imported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from .evaluation_memory import (
    MODEL_CALL_LOG,
    qwen_call_model,
    run_research_with_memory,
    utc_now,
)
from .llm import MODEL_NAME
from .planning import ResearchPlanner


def run_two_run_experiment(
    planner, ticker, research_question, call_model, *, memory_path, output_dir,
    execution_mode="unspecified",
):
    """Persist actual run artifacts and report whether the loop was exercised.

    Injectable planner/model support offline tests. Completion flags record
    execution behavior; they do not establish factual correctness or causality.
    """
    if execution_mode not in {"live", "offline_fixture", "unspecified"}:
        raise ValueError("Unknown execution_mode.")
    experiment_id = str(uuid4())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    files = []
    failure = None
    log_start = len(MODEL_CALL_LOG)
    try:
        for number in (1, 2):
            result = run_research_with_memory(
                planner, ticker, research_question, call_model,
                memory_path=memory_path,
            )
            result["execution_mode"] = execution_mode
            result["model"] = MODEL_NAME if execution_mode == "live" else None
            result["experiment_id"] = experiment_id
            result["run_number"] = number
            artifact = output / f"research_run_{result['run_id']}.json"
            artifact.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            runs.append(result)
            files.append(str(artifact))
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        log = {
            "experiment_id": experiment_id,
            "execution_mode": execution_mode,
            "model": MODEL_NAME if execution_mode == "live" else None,
            "finished_at": utc_now(),
            "completed_run_files": files,
            "failure": failure,
            "model_calls": MODEL_CALL_LOG[log_start:],
        }
        (output / f"model_calls_{experiment_id}.json").write_text(
            json.dumps(log, indent=2, allow_nan=False), encoding="utf-8",
        )
    new_lesson_used = any(
        lesson.get("source_run_id") == runs[0]["run_id"]
        for lesson in runs[1]["lessons_used"]
    )
    revisions_exercised = any(run["revisions_completed"] > 0 for run in runs)
    workflow_demonstrated = new_lesson_used and revisions_exercised
    analyses_accepted = all(run["passed"] for run in runs)
    return {
        "experiment_id": experiment_id,
        "execution_mode": execution_mode,
        "runs": runs,
        "run_files": files,
        "new_run_one_lesson_used_in_run_two": new_lesson_used,
        "revision_exercised": revisions_exercised,
        "workflow_demonstrated": workflow_demonstrated,
        "analyses_accepted": analyses_accepted,
        "demonstration_complete": (
            workflow_demonstrated and analyses_accepted
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the shared two-run GPU research workflow.")
    parser.add_argument("--live", action="store_true", help="Enable model downloads and live provider calls.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--question", default="Summarize market performance, financial metrics and news, with sources and limitations.")
    parser.add_argument("--memory", type=Path, default=Path("data/processed/research_lessons.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live from a configured CUDA GPU runtime to enable inference.")
    result = run_two_run_experiment(
        ResearchPlanner(), args.ticker, args.question, qwen_call_model,
        memory_path=args.memory, output_dir=args.output_dir, execution_mode="live",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
    print("Inspect drafts, citations, evaluations and lesson use manually; scores alone are not validation.")
    return 0 if result["demonstration_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
