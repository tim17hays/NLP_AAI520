# Final submission status — October 19, 2026

The combined code and final notebook are on `codex/final-team-integration`.
`main` has not been updated. Use the linked integration branch for Colab and
ensure any repository link in the submission identifies the correct branch.

## Final notebook and execution

- `notebooks/final_multi_agent_financial_analysis.ipynb` combines all three
  contributions, explanations, source evidence, checks, real-model recorded
  results, honest limitations, references and AI disclosure.
- Local execution is explicitly **recorded-evidence audit mode**: synthetic
  offline checks and the recorded Colab outputs are inspected without claiming
  a fresh GPU run.
- In a fresh Colab T4 GPU runtime, **Run all** enables new AAPL retrieval/Qwen
  drafting and Ashley's authored controlled two-run model experiment. The last
  cell attempts notebook/HTML capture and downloads an artifact ZIP.
- If Colab notebook capture is unavailable, download the executed `.ipynb` and
  use `python scripts/export_submission.py PATH_TO_NOTEBOOK` to create HTML.
- Check the rendered artifact and actual outputs before upload. Do not claim
  team approval, Canvas submission or a fresh GPU run from local audit alone.

## Correct interpretation of the completed demonstrations

Ashley's demonstrations show that a lesson was saved after Run 1 and retrieved
before Run 2. The recorded runs did not demonstrate improved analysis quality.

The controlled experiment used **real Qwen calls and fixed fictional DEMO
company evidence**, not fresh AAPL market data. Run 1 began with a deliberately
uncited human-written draft; Run 2 used the team Qwen generator. Those different
starting conditions prevent a causal claim that memory caused a quality change.

Recorded results: Run 1 stayed at 7/10 after two revisions and stopped at the
revision limit; Run 2 stayed at 5/10 after one completed revision and stopped on
an unchanged draft. Neither final analysis passed. Workflow/memory-delivery
checks passed. One process-lesson record was stored; returning that existing
record on Run 2 is not a second unique lesson.

Current citation checks reject missing/unknown references. Narrow benchmark
checks supplement the model evaluator; manual review still found unsupported
comparisons and incorrect calculations. Delivery is demonstrated; reliable
financial-answer improvement is not. These are documented limitations, not
fabricated passes or a missing demonstration.

## Code synchronization and ownership

Tim: retrieval/news, source/error trace and integration.
Roopleen: planning/routing and Qwen drafting.
Ashley: evaluator, bounded revisions, benchmark/citation checks and memory.

`src/evaluation_memory.py` now matches all 26 function bodies from Ashley's
updated authored notebook. `scripts/sync_evaluation_module.py` regenerates the
shared module without executing models or mounting Drive. A regression check
prevents the old shared-module drift from recurring. No evaluator algorithms or
prompts were redesigned during submission assembly.

The original component notebooks preserve their recorded outputs. Tim's
"Live handoff" wording and matching HTML are corrected. The old October 12
handoff and early October 18 audit are historical snapshots, not current results.
The final notebook is the canonical combined submission source.

## Remaining human steps

1. Tim runs the final notebook in a fresh Colab GPU runtime.
2. Inspect that run's drafts, evidence, lesson provenance, failure states and
   export. Fresh results may differ from the historical scores; report them as
   measured. A failed quality decision must remain failed.
3. Review the AI disclosure and collect final team approval.
4. Upload the approved PDF/HTML to Canvas and record confirmation. Each member
   submits their individual peer evaluation. Repository contents do not confirm
   those submissions or the official Canvas deadline.

No paid model-provider token balance is needed for this local open-weight model
setup. Colab GPU availability/runtime limits and model-download access still
apply. Optional news-provider keys stay private.

## Latest local checks

The synchronized code passes **31 tests**, with **2 opt-in live-provider tests
skipped**. All 26 evaluator function bodies match Ashley's authored notebook.
The final notebook's 14 code cells execute locally in recorded-evidence mode
without notebook errors. Its HTML preview is generated from those outputs.
This does not replace Tim's upcoming fresh Colab GPU run.
