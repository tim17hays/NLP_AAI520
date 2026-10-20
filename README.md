# Multi-Agent Financial Analysis System

AAI-520 final team project by Tim Hays, Roopleen Kaur and Ashley Marx.

## Final notebook

[Executed submission HTML](notebooks/final_multi_agent_financial_analysis.html) · [Executed notebook](notebooks/final_multi_agent_financial_analysis.ipynb)

[Open the final notebook in Google Colab](https://colab.research.google.com/github/tim17hays/NLP_AAI520/blob/main/notebooks/final_multi_agent_financial_analysis.ipynb).
Select **Run all**. The last cell exports the notebook as HTML and downloads a
submission ZIP. Submit the HTML file through the team assignment link.

The canonical release branch is `main`. The notebook
contains all four agent functions and all three workflow patterns, actual
experiment records, a same-input memory-policy comparison, source references,
GitHub links and a consolidated AI-assistance disclosure.

## Demonstrated behavior

- Planning selects ordered research steps and relevant specialists.
- Routing dynamically selects market, earnings, news or company tools.
- News chaining performs ingest, preprocessing, FinBERT classification,
  extraction and source-preserving summarization; actual FinBERT results and
  failure/provenance checks are displayed.
- Real Qwen critiques and independent guards identify output problems.
- The optimizer compiles a cited evidence brief using supplied values and
  arithmetic, with original model outputs retained separately.
- A persisted process lesson activates that policy before the second draft.
  The same raw candidate and evidence are compared with and without memory.

## Results and scope

The original Qwen2.5-1.5B Colab baseline failed acceptance: Run 1 stayed at 7/10
and Run 2 stayed at 6/10. A resource-bounded Qwen2.5-0.5B CPU reviewer also
rejected the corrected briefs. Those model judgments remain unchanged.

The evidence-tool outputs independently pass the documented arithmetic,
citation and constrained-wording checks. The stored lesson prevents the same
raw-draft errors before Run 2 evaluation. This demonstrates a bounded software
policy benefit, not general financial accuracy, model fine-tuning or successful
model acceptance. News and financial inputs used for controlled checks are
explicitly fictional. Real AAPL provider evidence is shown separately.

The default report execution validates recorded experiments without loading
model weights. Optional fresh experiments are available through
`scripts/run_grounded_demo.py` and `scripts/run_finbert_demo.py`. Actual model
identities, versions, source hashes, timestamps, raw judgments and review-cache
events are recorded. Scores from different reviewers are not directly compared
as a controlled model experiment.

## Team contributions

| Contributor | Primary work |
|---|---|
| Tim Hays | Retrieval/news, integration, evidence-tool policy and final report |
| Roopleen Kaur | Planning, specialist routing and Qwen draft generation |
| Ashley Marx | Evaluation, bounded revision and persistent process memory |

The original component records and Git history preserve authorship. Codex's
assistance, including authorized integration changes, is disclosed in the final
notebook.

## Source and validation

`src/` contains reusable modules; `tests/` contains offline regression checks;
`data/examples/` contains labeled inputs and actual recorded experiments.
See [final validation](docs/team/FINAL_READINESS.md). The checked source passed
66 offline tests; two opt-in live-provider tests were skipped. API credentials
remain in an untracked `.env` file and are not included in the submission.

Install dependencies with `pip install -r requirements.txt`. Optional data
provider keys are `FINNHUB_API_KEY` and `ALPHA_VANTAGE_API_KEY`; no paid LLM API
key is needed. Only one team representative uploads the final report. Peer
evaluations use the separate individual assignment link.
