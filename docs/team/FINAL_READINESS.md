# Final project validation record

The final release is `codex/final-team-integration`. The combined notebook is
`notebooks/final_multi_agent_financial_analysis.ipynb`; its last Colab cell
exports the final HTML and reproducibility ZIP.

## Rubric evidence

The notebook includes a requirement-to-evidence table and executable checks for
planning, selected tool use, self-reflection, persistent memory, news chaining,
specialist routing and evaluator–optimizer behavior. The source, traces,
experiments, explanations, comparison figure, GitHub link and AI disclosure are
included in one combined report.

## Actual findings

- Original Qwen2.5-1.5B Colab baseline: 7→7 and 6→6; both model-acceptance
  decisions remain false.
- Actual Qwen2.5-0.5B CPU ordinal review records are retained, including its
  rejection of the corrected brief. No score or threshold was changed to pass.
- The evidence tool computes 20% revenue growth, 10% operating margin and 10%
  closing-price change from supplied inputs, places valid references beside
  claims and quotes supplied news. Independent Decimal-based checks verify the
  named percentages; constrained wording and citation checks also pass.
- A Run 1 process lesson is saved to disk, retrieved before Run 2 planning and
  delivered to drafting. It activates the same evidence policy before the
  second draft. The same-input comparison shows the policy's bounded benefit.
- Actual FinBERT inference executes all five news stages and removes a
  duplicate. The fictional supplier-delay example is classified negative with
  confidence 0.8012; this confidence is not a probability of financial truth.
- Source fingerprints, raw model judgments, original baseline evidence and
  model/runtime identities are preserved. Recorded results are labeled as
  recorded; report execution does not claim fresh inference.

## Validation and limitations

66 offline tests passed; two optional live-provider tests were skipped. Tests
cover changed inputs, missing fields, source parity, stopping rules, citation
and benchmark guards, exact score provenance, memory identity, same-input
policy effects, parser recovery and retained failure records. Final notebook
source is JSON-valid and every code cell parses.

The evidence brief is schema-bound, not a universal fact checker. Passing its
checks does not change the model-acceptance flags, establish investment value
or guarantee an instructor's rubric rating. The report describes these limits
alongside the demonstrated behavior rather than hiding unsuccessful results.

## Submission

Run all in the linked final Colab notebook and submit the exported HTML. The
ZIP provides supporting artifacts. One team representative submits the team
report; individual peer evaluations are separate. The final narrative contains
no unresolved teammate handoffs or unfinished assembly instructions.
