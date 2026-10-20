"""Execute the evidence-tool memory demonstration with actual cached models.

Run from the repository: python -m scripts.run_grounded_demo --output DIR
The output directory must be new; no failed experiment is overwritten.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Loading model runtime", flush=True)
    import torch
    import transformers

    from src.criterion_reviewer import CriterionReviewer
    from src.evidence_verification import verify_demonstration
    from src.grounded_reporting import run_grounded_demonstration
    from src.llm import MODEL_NAME, _load_model

    source = Path("data/examples/grounded_baseline_input.json")
    baseline = json.loads(source.read_text())
    started = datetime.now(timezone.utc).isoformat()
    print("Loading cached Qwen weights", flush=True)
    tokenizer, model = _load_model()
    print(
        "Actual review model:", MODEL_NAME, "device:", model.device, flush=True
    )
    reviewer = CriterionReviewer(tokenizer, model, MODEL_NAME)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def persistent_review(instructions, payload):
        try:
            return reviewer(instructions, payload)
        finally:
            (output / "model_calls.json").write_text(
                json.dumps(reviewer.calls, indent=2)
            )

    try:
        report = run_grounded_demonstration(
            persistent_review,
            output,
            baseline,
            "fresh_local_models_tool_optimizer",
        )
    finally:
        if output.exists():
            (output / "model_calls.json").write_text(
                json.dumps(reviewer.calls, indent=2)
            )
    report["model_calls"] = reviewer.calls
    report["review_cache_events"] = reviewer.cache_events
    report["review_protocol"] = (
        "five separately prompted forced-choice ordinal model judgments; exact repeated inputs cached"
    )
    report["runtime"] = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model": MODEL_NAME,
        "device": str(model.device),
        "platform": platform.platform(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path("src/grounded_reporting.py"),
                Path("src/criterion_reviewer.py"),
                Path("src/evaluation_memory.py"),
                Path("src/llm.py"),
                Path("src/planning.py"),
            )
        },
    }
    (output / "demonstration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["checks"], indent=2), flush=True)
    print("Artifact:", output / "demonstration.json", flush=True)
    verification = verify_demonstration(report, baseline)
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2)
    )
    print(
        "Independent evidence verification:", verification["all_checks_passed"]
    )
    print(
        "Unchanged model acceptance:",
        verification["unchanged_model_acceptance"],
    )
    if not verification["all_checks_passed"]:
        raise SystemExit(
            "Evidence/workflow verification failed; inspect artifacts."
        )


if __name__ == "__main__":
    main()
