"""Build the final report from executed experimental artifacts and source code.

This edits explanatory text and cells, never experimental scores or responses.
Execute the resulting notebook and export it separately.
"""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'notebooks/final_multi_agent_financial_analysis.ipynb'


def main():
    notebook = nbformat.read(PATH, as_version=4)
    # Keep the original 30-cell report as a stable assembly source.
    if any(cell.get('id') == 'grounded-introduction' for cell in notebook.cells):
        raise SystemExit('The final sections already exist; avoid duplicate assembly.')

    def replace(index, source):
        notebook.cells[index].source = source.strip() + '\n'

    replace(0, '''# Multi-Agent Financial Analysis System
**AAI-520 Final Team Project**
**Tim Hays · Roopleen Kaur · Ashley Marx**
[GitHub repository](https://github.com/tim17hays/NLP_AAI520) · [Final project branch](https://github.com/tim17hays/NLP_AAI520/tree/codex/final-team-integration)

This final report documents planning, specialist tool use, self-reflection and persistent memory, together with news chaining, routing and evaluator–optimizer workflows. It includes actual AAPL provider/model evidence, real Qwen baseline failures, an evidence-tool correction strategy, and an observed two-run memory-policy demonstration. Model acceptance and evidence verification are reported as separate outcomes.

**How to read the results:** The original free-form Qwen baseline failed acceptance in both runs. The completed hybrid demonstration uses Qwen to critique and score, Python to calculate and cite a bounded evidence brief, and a saved process lesson to activate the safer policy before the second draft. Model judgments, tool-generated text, scripted tests and recorded experiments are labeled separately. The results establish a software-policy benefit on fixed fictional evidence, not general financial accuracy, model fine-tuning or an investment recommendation.

**Reproducibility:** Run all reproduces this report and validates the supplied experiment records without downloading model weights. The model-execution commands and exact recorded runtime details appear below. Fresh provider/model experiments are optional reproductions, not prerequisites for reading this completed submission.''')
    replace(1, '''## 1. Design, ownership and rubric evidence
| Requirement | Implementation | Visible evidence |
|---|---|---|
| Plans research | `ResearchPlanner.plan()` | Section 4: selected specialists, ordered steps and lesson-review step |
| Uses tools dynamically | Research-question routing selects tools | Section 4: market/earnings/news/fallback routes; AAPL tool and error trace |
| Self-reflects | Real Qwen critique plus citation/content guards | Sections 7–8: original judgments, failures, re-evaluation and acceptance |
| Learns across runs | Saved process lesson activates a fixed safer policy | Section 8: disk record, lesson identity, next plan and draft; same-input ablation |
| Prompt chaining | Ingest → preprocess → classify → extract → summarize | Section 3: trace, deduplication, actual FinBERT inference and source alignment |
| Routing | Market, earnings, news and company specialists | Section 4: different questions select different specialist tools |
| Evaluator–optimizer | Critique → evidence-tool revision → re-evaluation | Section 8: failed original, repaired brief and subsequent real model judgment |
| Code and communication | Reusable modules, tests, GitHub, comments, diagrams and disclosure | Sections 1–10 and references |

Tim owns retrieval/news and integration; Roopleen owns planning/routing and initial draft generation; Ashley owns evaluation, revision and persistent memory. The final integration adds an explicitly labeled evidence-tool optimizer and keeps their component implementations and historical experiment records traceable in Git.

The architecture combines LLM judgment with deterministic routing and arithmetic. Routing is a documented keyword heuristic. Memory changes an auditable drafting policy; it does not update model weights. The bounded numeric brief is applicable only when its required compatible inputs, units and periods are supplied.''')
    replace(3, '''## 2. Reproducible runtime and repository
The original generation/review model is `Qwen/Qwen2.5-1.5B-Instruct`; the resource-bounded local reviewer is `Qwen/Qwen2.5-0.5B-Instruct`; news sentiment uses `ProsusAI/finbert`. Both run through Hugging Face Transformers without a paid LLM API key. Optional Finnhub/Alpha Vantage credentials remain outside the repository.

The report displays the actual Colab baseline and local model-experiment metadata. Its default execution validates and presents those recorded results; this does not claim a new model call on every report execution. The source and experiment artifacts are included in the final branch. The loader supports CUDA, Apple MPS and CPU; CPU execution is slower.

To repeat the experiments with model weights installed, run `python -m scripts.run_grounded_demo --output outputs/NEW_EXPERIMENT_DIRECTORY` and `python -m scripts.run_finbert_demo --model-path PATH_TO_FINBERT --output outputs/finbert_demo.json`. A new directory prevents memory contamination. `AAI520_DEVICE` selects a device; `AAI520_MODEL_PATH` can identify a local cached Qwen snapshot.''')
    source = notebook.cells[4].source
    source = source.replace('RUN_LIVE_MODELS = IN_COLAB and os.getenv("AAI520_RECORDED_ONLY") != "1"', 'RUN_LIVE_MODELS = IN_COLAB and os.getenv("AAI520_FRESH_BASELINE") == "1"')
    source = source.replace('"recorded-evidence audit; no new model inference"', '"final report: verified recorded experiments and executable checks"')
    replace(4, source)
    replace(6, '''## 3. News prompt chain and source validation — Tim
The production chain ingests news, normalizes text and URLs, deduplicates records, classifies retained text with FinBERT, extracts source sentences and produces a source-preserving summary. Every stage leaves a visible trace.

The first checks use a deterministic injected classifier to isolate provenance and error behavior. The subsequent experiment uses actual FinBERT inference on explicitly fictional classroom news; its input is duplicated to test preprocessing. Neither classification confidence nor a source quotation establishes the truth of an article or predicts a stock return.

Yahoo news requests have a 20-second per-request timeout and a 15-minute successful-result cache. The fallback order is Yahoo → Finnhub → Alpha Vantage. Historical-price requests use a 15-second timeout. These are per-request bounds, not an overall workflow deadline.''')
    replace(13, '''### AAPL provider evidence and Qwen draft
This section preserves the AAPL provider/model result from the completed Colab execution associated with commit `34dbce9`. Its source URLs, access times, missing values and errors remain visible. It demonstrates real provider integration separately from the fixed fictional benchmark used to test correction and memory.''')
    source = notebook.cells[14].source
    source = source.replace('REPO / "notebooks/person2_aapl_live_handoff.json"', 'REPO / "data/examples/final_aapl_colab_result.json"')
    source = source.replace("Roopleen's recorded provider/model execution; not refreshed locally", 'Recorded fresh Colab provider/model evidence, commit 34dbce9')
    replace(14, source)
    replace(15, '''Source-attributed retrieval does not guarantee a grounded generated narrative. The AAPL draft is retained as model output and must be interpreted against the displayed source records. The controlled correction experiment below evaluates specific known errors without pretending that a model score independently verifies real-world financial claims.''')
    replace(20, '''## 6. Original real-model baseline
The original Colab experiment used fixed fictional DEMO evidence. Run 1 began with an uncited benchmark draft; Run 2 used Qwen generation after receiving a saved lesson. Both evaluation/revision histories are retained, including failed outcomes.

The different starting draft origins prevent a causal comparison of memory from these two baseline scores. Section 8 therefore holds the raw candidate and evidence fixed when testing the new memory-controlled drafting policy. The baseline remains a record of free-form model behavior, not evidence of successful correction.''')
    source = notebook.cells[23].source.replace('REPO / "data/examples/recorded_controlled_results.json"', 'REPO / "data/examples/final_colab_baseline.json"')
    source = source.replace("ASHLEY'S RECORDED REAL-MODEL EXECUTION; NOT A NEW LOCAL GPU RUN", 'RECORDED FRESH COLAB BASELINE — OCTOBER 19, 2026, COMMIT 34dbce9')
    replace(23, source)
    replace(24, '''### Baseline findings and design response
In the completed October 19 Colab baseline, Run 1 stayed at **7/10** after two revisions. Run 2 stayed at **6/10** and stopped when its proposed revision was unchanged. Neither final analysis passed, although all workflow and lesson-delivery checks passed.

Run 1 omitted citations and incorrectly described an unadjusted price change as total return. Run 2 invented revenue of $80 million instead of the supplied $120 million, reported −20% revenue growth instead of +20%, and treated a $4 closing-price increase as 4%. The correct supplied calculations are **20% revenue growth, 10% operating margin and 10% price change**. The saved lesson reached the generator but did not reliably control its wording.

The design response is an evidence-bound optimizer, not a weaker passing threshold. It computes the supported metrics and retains cited source text, then submits its candidate to the existing evaluation contract and deterministic guards. The bounded CPU reviewer uses separately prompted ordinal judgments to avoid JSON serialization failures; each score is selected by the model and logged. The model still rejects the corrected brief, so independent evidence verification is reported separately rather than changing its acceptance flags. A retrieved process lesson activates that same fixed drafting policy before a later run. This is explicitly tool-assisted generation, not a repaired answer relabeled as an original LLM response.''')
    replace(26, '''## 9. Conclusions and submission record
The final project documents all four agent functions and all three workflow patterns through the source, traces, experiments and checks displayed above. The functional checklist reports what executed rather than predicting a rubric score.

The original Qwen failures remain visible. The evidence-tool experiment tests a specific architectural remedy: real model critique, a cited arithmetic brief, re-evaluation, and memory-controlled preflight drafting. The same-input comparison isolates the software-policy effect on this fixed case. It does not establish general financial reliability or a causal improvement in the underlying language model.

The final submission is this executed notebook rendered as HTML, with the GitHub branch, source records, exact runtime metadata, references and AI disclosure. The accompanying reproducibility ZIP preserves the notebook and supporting artifacts. Canvas upload is performed by the team representative; individual peer evaluations use the separate assignment link.''')
    replace(28, '''## 10. AI assistance disclosure and responsibility
Tim used OpenAI Codex to assist with retrieval reliability, repository integration, synchronization of authored functions, evaluator/revision prompts and payloads, model-response formatting recovery, the evidence-tool optimizer, memory-policy demonstrations, regression tests, notebook assembly, narrative revision and export validation. These contributions include authorized changes to Ashley's shared evaluator/revision integration. The project documents the resulting code, design choices, tests and limitations rather than presenting unexamined AI output as verified research.

Roopleen's planning/routing/Qwen work and Ashley's evaluator/revision/memory work retain their authorship and Git history. Ashley's component notebook includes her own AI-use disclosure. Actual Colab and local model runs are identified by their recorded runtime and provenance. Tool-generated briefs, model judgments, source quotations and scripted test fixtures are distinguished throughout.

The evidence brief intentionally uses deterministic arithmetic and source-bound wording to address observed model failures; it is not claimed as an LLM-generated improvement. The team remains responsible for understanding, explaining and submitting its work. This notebook does not assert an unverified team approval or a Canvas submission.''')

    def md(id_, text):
        cell = nbformat.v4.new_markdown_cell(text.strip() + '\n')
        cell.id = id_
        return cell

    def code(id_, text):
        cell = nbformat.v4.new_code_cell(text.strip() + '\n')
        cell.id = id_
        return cell

    finbert_cells = [
        md('finbert-observation', '''### Actual FinBERT chaining experiment
The artifact below was produced by running the real `ProsusAI/finbert` model through the production `NewsPipeline` on duplicated fictional source text. It shows the complete five-stage trace, removes the duplicate and retains the source alignment. This is an execution example, not a measured sentiment-accuracy benchmark.'''),
        code('finbert-trace', '''news_model_report = json.loads((REPO / "data/examples/final_finbert_demonstration.json").read_text())
assert news_model_report["execution_mode"] == "fresh_local_finbert"
news_model_result = news_model_report["result"]
assert news_model_result["workflow"] == ["ingest", "preprocess", "classify_finbert", "extract", "summarize"]
assert not news_model_result["errors"]
assert news_model_result["trace"]["ingest"]["input_count"] == 2
assert news_model_result["trace"]["preprocess"]["retained_count"] == 1
print("Actual model runtime:", json.dumps({key: value for key, value in news_model_report.items() if key not in ("result", "raw_news")}, indent=2))
print("Input:", json.dumps(news_model_report["raw_news"], indent=2))
print("Five-stage output:", json.dumps(news_model_result["trace"], indent=2))
print("Source-attributed summary:", news_model_result["summary"])
print("PASS: actual FinBERT inference, deduplication and all five workflow stages.")'''),
    ]
    grounded_cells = [
        md('grounded-introduction', '''## 8. Evidence-tool optimization and learning across runs
This completed experiment keeps one previously generated Qwen candidate and its fictional evidence fixed. The actual `ResearchPlanner` selects and executes dataset-backed market, earnings and news specialists. The existing evaluator/memory loop receives real Qwen ordinal judgments; an explicit deterministic evidence tool handles correction instead of asking the same small model to rewrite unsupported numbers again.

**Run 1:** evaluate the original candidate, compile a cited brief after failure, re-evaluate that new candidate, independently verify its evidence-bound calculations and citations, and save the observed process lesson to disk. **Run 2:** retrieve that record before planning and drafting; the lesson activates the same evidence-bound policy before evaluation. The model still assigns the scores and the existing acceptance thresholds remain unchanged. The local CPU reviewer is Qwen2.5-0.5B-Instruct; original Colab baseline scores used Qwen2.5-1.5B-Instruct, so those two experiments are not a controlled model-to-model score comparison. The same-input policy ablation is deterministic and does not depend on changing reviewers.

The report compiler reads evidence fields, units, periods and formulas. It does not read a gold/reference answer or insert DEMO-specific percentages. Tests change the inputs and verify changed results. Missing inputs or unresolved snapshot errors cause a clear failure instead of invented facts.'''),
        code('grounded-implementation', '''from src.grounded_reporting import (
    EvidenceBriefDraft, ToolAssistedReviewer, compile_evidence_brief,
    run_grounded_demonstration,
)
print("Evidence-tool compiler:\\n", inspect.getsource(compile_evidence_brief))
from src.criterion_reviewer import CriterionReviewer
print("Ordinal model reviewer:\\n", inspect.getsource(CriterionReviewer))
print("Optimizer boundary:\\n", inspect.getsource(ToolAssistedReviewer))
print("Memory-controlled drafting:\\n", inspect.getsource(EvidenceBriefDraft))'''),
        code('grounded-results', '''import hashlib
grounded = json.loads((REPO / "data/examples/final_grounded_demonstration.json").read_text())
assert grounded["execution_mode"] == "fresh_local_models_tool_optimizer"
verification = json.loads((REPO / "data/examples/final_grounded_verification.json").read_text())
assert verification["all_checks_passed"], verification["checks"]
assert hashlib.sha256((REPO / "src/evidence_verification.py").read_bytes()).hexdigest() == verification["verifier_sha256"]
for relative, expected_hash in grounded["runtime"]["source_hashes"].items():
    assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected_hash, relative
print("Actual experiment runtime:", json.dumps(grounded["runtime"], indent=2))
print("Draft provenance:", json.dumps(grounded["baseline_provenance"], indent=2))
rows = []
for number, run in enumerate(grounded["results"], 1):
    rows.append(f"<tr><td>{number}</td><td>{run['initial_score']}/10</td><td>{run['revised_score']}/10</td><td>{run['revisions_completed']}</td><td>{run['passed']}</td><td>{len(run['lessons_used'])}</td></tr>")
display(HTML("<table><thead><tr><th>Run</th><th>Initial</th><th>Final</th><th>Tool revisions</th><th>Accepted</th><th>Lessons received</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"))
print("Workflow and independent evidence checks:", json.dumps(verification["checks"], indent=2))
print("Unchanged model acceptance:", verification["unchanged_model_acceptance"])
print("Independent final evidence verification:", json.dumps(verification["final_evidence_checks"], indent=2))
print("Review protocol:", grounded["review_protocol"])
print("Exact-input review cache:", json.dumps(grounded["review_cache_events"], indent=2))
for number, run in enumerate(grounded["results"], 1):
    print(f"\\nRUN {number} INITIAL DRAFT\\n", run["initial_draft"])
    print("FINAL ANALYSIS\\n", run["revised_analysis"])
    print("REAL EVALUATION HISTORY\\n", json.dumps(run["revision_history"], indent=2))
print("Explicit tool repairs:", json.dumps(grounded["tool_repairs"], indent=2))
print("Raw model responses:", json.dumps([{key: call.get(key) for key in ("call_id", "task", "status", "response_text", "format_normalization", "error")} for call in grounded["model_calls"]], indent=2))'''),
        code('memory-disk-ablation', '''import tempfile
from src.evaluation_memory import save_lesson, retrieve_lessons
fixed = json.loads((REPO / "data/examples/grounded_baseline_input.json").read_text())
with tempfile.TemporaryDirectory(prefix="aai520-final-memory-") as directory:
    memory_path = Path(directory) / "lessons.json"
    saved = grounded["results"][0]["saved_lesson"]
    save_lesson(saved, memory_path)
    retrieved = retrieve_lessons("DEMO", memory_path=memory_path)
    assert any(item["lesson_id"] == saved["lesson_id"] for item in retrieved)
    drafting = EvidenceBriefDraft(fixed["candidate"])
    arguments = dict(ticker="DEMO", research_question=fixed["context"]["research_question"], evidence=fixed["context"]["evidence"])
    without_memory = drafting(**arguments, lessons=[])
    with_memory = drafting(**arguments, lessons=retrieved)
    assert without_memory == fixed["candidate"]
    assert with_memory == grounded["results"][1]["initial_draft"]
    assert drafting.calls[0]["policy_applied"] is False
    assert drafting.calls[1]["policy_applied"] is True
    print("Lesson actually saved and read from disk:", json.dumps(retrieved, indent=2))
    print("Run 1 source identity:", saved["source_run_id"])
    print("Run 2 received lesson identity:", grounded["results"][1]["lessons_used"][0]["lesson_id"])
print("Same-input comparison:", json.dumps(grounded["same_input_ablation"], indent=2))
first, second = grounded["results"]
display(Markdown(f"**Observed interpretation:** Run 1 changed from {first['initial_score']}/10 to {first['revised_score']}/10 after {first['revisions_completed']} tool revision(s); Run 2 began at {second['initial_score']}/10 and completed {second['revisions_completed']} changed revisions and {second['revision_calls']} optimizer attempt(s). The exact same raw candidate and evidence produce failed citation/benchmark checks without the stored lesson and passing checks under the fixed evidence policy. This demonstrates retained-state use and a bounded policy benefit, not improved model weights or general financial accuracy."))'''),
        code('memory-policy-figure', '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ablation = grounded["same_input_ablation"]
labels = ["Citation failures", "Narrow benchmark issues"]
before = [len(ablation["without_memory"]["citation_check"]["issues"]), len(ablation["without_memory"]["benchmark_check"]["issues"])]
after = [len(ablation["with_memory"]["citation_check"]["issues"]), len(ablation["with_memory"]["benchmark_check"]["issues"])]
x = list(range(len(labels)))
figure, axis = plt.subplots(figsize=(8, 4))
for offset, values, label, color in [(-0.18, before, "Without stored lesson", "#b04b3b"), (0.18, after, "With lesson-driven evidence policy", "#2f7e71")]:
    bars = axis.bar([value + offset for value in x], values, width=0.36, label=label, color=color)
    axis.bar_label(bars, padding=3)
axis.set_xticks(x, labels)
axis.set_ylabel("Observed issue count")
axis.set_title("Same candidate and evidence before revision")
axis.set_ylim(0, max(before + after) + 1)
axis.yaxis.get_major_locator().set_params(integer=True)
axis.legend(loc="upper right", fontsize=8)
figure.tight_layout()
figure.savefig(OUTPUT_DIR / "memory_policy_comparison.png", dpi=180)
display(figure)
plt.close(figure)
print("Counts describe this fixed benchmark and its narrow checks, not general accuracy.")'''),
        md('grounded-scope', '''### What the evidence-policy demonstration establishes
The model-based reviewer, independent citation/content guards, computed evidence ledger, saved lesson and next-run policy decision are visible separately. The CPU reviewer rated even the corrected brief poorly; both legacy model-acceptance flags remain false. The positive result is independently verified correction of the supplied metrics and citations, followed by prevention of the same raw-draft errors under the saved policy. No model score or acceptance rule was changed to manufacture a pass. The corrected brief includes 20% revenue growth, 10% operating margin and 10% closing-price change with compatible periods and citations. News is quoted as supplied rather than promoted to a verified real-world event. Price change is explicitly distinguished from total return.

The evidence-tool optimizer is intentionally conservative and schema-bound. It rejects incomplete inputs instead of guessing reporting periods or financial units. A correct result on the fictional benchmark does not establish broad factual verification, predictive investment value or robustness to every real provider schema. The original AAPL example and failed Qwen baseline show why these limits matter.'''),
        code('final-functional-checklist', '''functional_checks = {
    "Plans research": set(TEAM_PLANNER.plan("AAPL", "market financial news").specialists) == {"market", "earnings", "news"},
    "Uses selected tools": len(grounded["results"][0]["tool_trace"]) == 3 and bool(AAPL_RESULT["tool_trace"]),
    "Self-reflects": bool(grounded["results"][0]["revision_history"][0]["critical_issues"]) and len(grounded["model_calls"]) >= 3,
    "Learns across runs": grounded["checks"]["run_2_drafting_policy_activated"] and grounded["checks"]["same_input_ablation_repaired_benchmark"],
    "Prompt chaining": len(news_model_result["workflow"]) == 5 and not news_model_result["errors"],
    "Routing": TEAM_PLANNER.plan("AAPL", "earnings and revenue").specialists == ["earnings"] and TEAM_PLANNER.plan("AAPL", "recent news sentiment").specialists == ["news"],
    "Evaluator-optimizer": verification["final_evidence_checks"][0]["passed"] and grounded["results"][0]["revisions_completed"] > 0,
}
assert all(functional_checks.values()), functional_checks
print(json.dumps(functional_checks, indent=2))
print("All seven agent-function/workflow requirements have visible executable evidence.")
print("This checklist validates demonstrated behavior; it does not predict the instructor's grade.")'''),
    ]
    # Work backwards to retain stable source indices while inserting sections.
    notebook.cells[26:26] = grounded_cells
    notebook.cells[11:11] = finbert_cells
    # Update the final manifest cell after inserting the new sections.
    final = notebook.cells[-1].source
    final = final.replace('"quality_checks": observed["quality_checks"], "manual_findings": manual_findings}', '"quality_checks": observed["quality_checks"], "manual_findings": manual_findings,\n            "grounded_checks": grounded["checks"], "independent_evidence_verification": verification, "functional_checks": functional_checks,\n            "grounded_runtime": grounded["runtime"], "baseline_provenance": observed.get("provenance"),\n            "report_mode": "completed_experiment_records_and_executable_verification"}')
    final = final.replace('Local audit complete. No new GPU results or Canvas submission are claimed.', 'Final report verification complete. Model experiment provenance is recorded above; Canvas submission is a separate action.')
    notebook.cells[-1].source = final
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []
    notebook.metadata['final_report'] = {
        'artifact_scope': 'actual Colab baseline, actual local model critique, evidence-tool optimization and memory-policy ablation',
        'authors': ['Tim Hays', 'Roopleen Kaur', 'Ashley Marx'],
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, PATH)
    print('Final report source assembled:', PATH)


if __name__ == '__main__':
    main()
