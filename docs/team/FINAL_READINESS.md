# Shared integration status

Combined branch: `codex/final-team-integration`. Publishing succeeded; the GitHub
tool request to create a pull request was rejected, so `main` has not been updated.
Review/merge this branch to make the shared code available on `main`.

All three authors' branches are combined into this version, preserving their
commit history and original notebooks. This integration connects existing work;
it does not claim that the final submission or live feedback validation is done.

## Canonical code and entry point

- `src/data_tools.py`, `src/news_pipeline.py`, `src/alternative_sources.py`: Tim's retrieval and five-stage news chain.
- `src/planning.py`, `src/llm.py`: Roopleen's planner/Qwen work with Tim's lesson/source-preservation integration fixes.
- `src/evaluation_memory.py`: Ashley's evaluator, revision and persistent-memory functions extracted from her notebook without changing scoring or prompts. Importing this module does not mount Drive or load models.
- `src/workflow.py`: shared two-run driver, artifact logging and explicit loop-completion flags.
- `notebooks/team_integration_runner.ipynb`: common setup and actual workflow calls for team GPU testing. No saved live results are claimed for this new runner.

The original component notebooks retain their historical outputs and pinned
checkouts. Use the shared runner to test the current combined code, rather than
copying the old notebooks' repository setup cells into a combined notebook.

## Validation boundaries

Offline tests verify wiring, failure/stopping behavior, source preservation,
lesson persistence and delivery using clearly labeled scripted fixtures.
`tests/test_team_workflow.py` exercises the actual shared planner with Ashley's
evaluation/revision/memory implementation and all ten of her acceptance groups.
CI runs lightweight offline checks without weights or provider calls.

Recorded real Colab results are retained in the original notebooks. Ashley's
recorded live runs both scored 10/10 but made zero revisions and saved no new
lesson. Her notebook acknowledges missing citations were incorrectly accepted.
The scripted offline lesson example does not establish actual LLM improvement.
The new shared driver does not conceal this: its demonstration flag is false
unless a revision occurs, a new run-one lesson reaches run two, and both final
evaluations pass. That flag still does not verify factual correctness or prove
that memory caused an improvement. Check citations and claims manually.

## Still required before submission

| Work | Owner | Evidence needed |
| --- | --- | --- |
| Tighten evaluator acceptance for missing citations and unsupported claims | Ashley | Real flawed-draft critique, correction and re-evaluation |
| Exercise persistent memory with the actual model | Ashley + Roopleen | Saved run-one lesson, retrieval before run two and inspected effect |
| Assemble the final team notebook | Roopleen, all contribute | One narrative with all workflows, functions, code, outputs and GitHub link |
| Fresh GPU execution of the combined notebook | Ashley + Roopleen; Tim reviews | Complete ordered outputs, source dates, runtime/model versions and honest failures |
| Export readable PDF or HTML | Ashley | Visual QA and all three approvals |
| Submit team artifact and individual peer evaluations | Tim submits team artifact; each submits own evaluation | Canvas confirmation; not inferable from GitHub |

Proposed submission notebook: `notebooks/final_multi_agent_financial_analysis.ipynb`.
It does not exist yet. The shared runner is an integration tool, not a final
submission. Tim's HTML is a component handoff, not a full team export.

## Authorship and AI assistance

Original planning/Qwen work is Roopleen Kaur's, evaluator/revision/memory work is
Ashley Marx's, and retrieval/news and integration work are Tim Hays's. Codex
assisted Tim with consolidation, extraction of existing functions, shared runner,
tests and documentation. Review and adapt the AI disclosure in the final artifact.

`INTEGRATION_HANDOFF.md` and `FINAL_READINESS_AUDIT_2026-10-18.md` are dated
historical snapshots. This file and the root README describe the combined code.
Private reply drafts, API keys, local memories and generated outputs remain
untracked. Do not put secrets into notebooks or JSON example artifacts.

## Consolidation checks

The combined suite passes **30 tests**, with **2 live-provider tests skipped**.
The shared runner's code cells compile; they have not been executed with real
models for this combined commit. Ashley's extracted function bodies match her
notebook source. All three remote team branch tips are preserved in the merge
history. Syntax and whitespace checks pass. The private message draft and `.env`
are excluded from tracked files.
