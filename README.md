# Multi-Agent Financial Analysis System

AAI-520 final team project by Tim Hays, Roopleen Kaur, and Ashley Marx.

The project implements a multi-agent workflow for a single-stock research run. The system will retrieve market and financial data plus news, plan and route research tasks, produce an evidence-based draft analysis, evaluate and revise the draft, and save reusable lessons for a later run.

## Team roles

| Team member | Primary responsibilities |
| --- | --- |
| Tim Hays | Team coordination, GitHub, data and news retrieval, prompt-chaining workflow |
| Roopleen Kaur | Research planning, routing, specialist integration, notebook assembly |
| Ashley Marx | Evaluation, revision loop, memory, final export and layout review |

## Research workflow

1. Accept a stock ticker and research request.
2. Plan the work and route requests to market, financial, earnings, and news specialists.
3. Retrieve and normalize source evidence.
4. Draft a sourced financial analysis.
5. Evaluate the draft, revise it within a bounded iteration limit, and retain a useful lesson for a future run.

## Repository layout

```text
src/          Reusable agent, retrieval, routing, evaluation, and memory code
notebooks/    Final project notebook and exploratory work
data/examples/ Small, non-sensitive sample inputs only
tests/        Unit and integration tests
docs/         Architecture notes and source documentation
```

## Person 1 data and news handoff

`src/data_tools.py` provides source-attributed company, market, and financial
snapshots through Yahoo Finance (`yfinance`). `src/news_pipeline.py` implements
the required news chain: ingest, preprocess, FinBERT classify, extract evidence,
and create an extractive, source-preserving summary. Both modules return
JSON-compatible payloads so the planner and routing components can consume them
without relying on pandas or provider-specific objects.

The default demonstration ticker is `AAPL`. The live FinBERT model
(`ProsusAI/finbert`) downloads on first use; tests use a deterministic classifier
fixture and do not call external services.

### Reliable live-data fallbacks

Yahoo Finance remains the default. When its news endpoint is empty or
rate-limited, Person 1 can use either independent, key-based provider below.
Both return the same `ToolResult` and `raw_news` structures as the Yahoo tools,
so Person 2 does not need a provider-specific route.

| Provider | Capabilities | Required environment variable |
| --- | --- | --- |
| Alpha Vantage | Quote, company overview, market news/sentiment | `ALPHA_VANTAGE_API_KEY` |
| Finnhub | Quote, basic financial metrics, company news | `FINNHUB_API_KEY` |

Add only the key(s) you choose to your untracked `.env` file. Each API response
retains its provider name, URL, and access timestamp. The final notebook must
state data access dates, source limitations, and that the output is educational
research rather than personalized financial advice.

The Yahoo adapter calls Yahoo Finance's standard GET search endpoint once with
a 20-second timeout and keeps a successful response in a 15-minute in-memory
cache. This avoids yfinance's cookie/crumb POST path for news retrieval while
the market-data tools continue to use yfinance. It does not attempt to
circumvent Yahoo's limits: if Yahoo returns no items, a rate-limit error, or a
timeout, the pipeline uses its fallback providers instead of repeatedly
retrying Yahoo.

The integrated news pipeline now performs that fallback automatically: Yahoo is
attempted once, then Finnhub and Alpha Vantage are tried in that order when
their keys are present. Market-data tools also reuse the same yfinance ticker
client within a run and apply a 15-second timeout to historical-price requests.

## Setup

Create a virtual environment, install dependencies, then add required API keys to a local `.env` file. Do not commit API keys, large raw datasets, or generated notebook checkpoints.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Shared team version

The combined work is on `codex/final-team-integration`, pending merge into `main`.
The integration runner clones that branch until the merge is approved.

The combined repository preserves Tim's, Roopleen's and Ashley's commits and
original component notebooks. **The final combined notebook is assembled. Tim must run it in a fresh Colab GPU runtime and review the executed export before submission.**
See [current readiness and ownership](docs/team/FINAL_READINESS.md).

| Component | Shared code | Author / integration |
| --- | --- | --- |
| Data and five-stage news chain | `src/data_tools.py`, `src/news_pipeline.py`, `src/alternative_sources.py` | Tim |
| Planning, routing and Qwen draft | `src/planning.py`, `src/llm.py` | Roopleen, with Tim's lesson/source handoff fixes |
| Evaluation, bounded revision and persistent memory | `src/evaluation_memory.py` | Ashley, extracted from her original notebook |
| Shared two-run workflow and artifact logging | `src/workflow.py` | Team interface, integrated by Tim |

### Run the current combined workflow

Use Python 3.12 and a CUDA GPU runtime for actual Qwen inference. The model is
`Qwen/Qwen2.5-1.5B-Instruct` through Hugging Face Transformers, with
`ProsusAI/finbert` for news sentiment. No paid LLM API key is required. Model
weights download on first use. Optional data-provider keys stay in `.env`.

Start with [the final combined notebook](notebooks/final_multi_agent_financial_analysis.ipynb)
in a fresh Colab T4 GPU runtime and choose **Run all**. It includes the full
research flow, authored controlled experiment, actual historical results,
limitations, references and export/download cell. The shorter
[team integration runner](notebooks/team_integration_runner.ipynb) remains a
component debugging tool. To run live AAPL research from the repository root:

```bash
python -m src.workflow --live --ticker AAPL
```

The driver retrieves lessons before each planner call, uses the actual draft
once, evaluates/revises, saves a valid lesson when eligible, and repeats for run
two. It records both run JSON files and the evaluation model-call log in ignored
`outputs/`. Memory is stored in ignored `data/processed/research_lessons.json`.
Use `--memory PATH` for a persistent Drive location in Colab; runtime-local
storage is temporary. Do not delete existing lessons to make a demonstration pass.

Exit status 0 means both final evaluations passed, a revision was exercised, and
a new run-one lesson reached run two. Workflow delivery and quality acceptance
are reported separately. Exit status 2 after completed runs means one or more
acceptance/delivery conditions were not met, not that no lesson could be saved. Runtime/model/provider failures remain errors.
These checks do not certify factual correctness or prove memory caused an
improvement: inspect source citations, evidence, feedback and the actual drafts.

### Call individual components

```python
from src.planning import ResearchPlanner
from src.evaluation_memory import qwen_call_model, run_research_with_memory

result = run_research_with_memory(
    planner=ResearchPlanner(),
    ticker="AAPL",
    research_question="Summarize market performance, financial metrics and news with sources.",
    call_model=qwen_call_model,
    memory_path="data/processed/research_lessons.json",
)
print(result["revised_analysis"])
print(result["revision_history"])
print(result["saved_lesson"])
```

For drafting alone, `ResearchPlanner().execute(ticker, research_question,
lessons=...)` returns `draft_analysis`, `evidence`, `plan`, `retrieved_lessons`
and `tool_trace`.

### Component evidence and final notebook

- [Tim: data/news notebook](notebooks/tim_data_news_handoff.ipynb) and
  [component HTML](notebooks/tim_data_news_handoff.html).
- [Roopleen: planner/Qwen Colab handoff](notebooks/roopleen_planner_qwen_team_handoff.ipynb)
  and [recorded AAPL result](notebooks/person2_aapl_live_handoff.json).
- [Ashley: evaluation/revision/memory notebook](notebooks/evaluation_revision_memory.ipynb).

Original notebooks preserve their historical outputs and setup pins; use the
shared runner for new tests of this combined code. They are not three separate
final submissions. The final combined source is
[final_multi_agent_financial_analysis.ipynb](notebooks/final_multi_agent_financial_analysis.ipynb).
Local validation uses clearly labeled recorded-evidence mode; Colab enables
fresh model calls by default. Download the executed notebook and export HTML
with the last cell or `python scripts/export_submission.py PATH_TO_NOTEBOOK`.

Ashley's demonstrations show that a lesson was saved after Run 1 and retrieved
before Run 2. The recorded runs did not demonstrate improved analysis quality.
The controlled real-model results were 7/10 to 7/10 (two revisions) and 5/10 to
5/10 (one completed revision); neither final analysis passed. Process-lesson
delivery succeeded. Fixed fictional evidence, different initial draft origins,
manual factual errors and citation limitations are explicitly documented. Do
not replace these failures with claims of a successful quality improvement.

## Tests

```bash
python -m pytest -q
```

Ordinary tests use injected provider/model fixtures and make no weight downloads
or live requests. GitHub Actions runs these offline checks, including all ten
of Ashley's acceptance groups and the actual planner-to-evaluator two-run
interface. This verifies plumbing, not actual model quality or learned behavior.

Live provider tests are opt-in and depend on provider availability:

```bash
AAI520_LIVE_TESTS=1 python -m pytest -q
```

Use the shared GPU runner for actual model integration testing. Review the
[readiness checklist](docs/team/FINAL_READINESS.md) before final export.

## Final Colab and export steps

1. Open the final notebook from `codex/final-team-integration` in Colab.
2. Select a T4 GPU runtime, then **Run all**. Actual model weights are downloaded
   on first use; the code prints the repository revision and runtime versions.
3. Save/download the executed notebook and artifact ZIP. The ZIP contains model
   call logs and reports; if automatic notebook capture fails, export the
   downloaded notebook locally with `scripts/export_submission.py`.
4. Review all outputs, limitations and the disclosure, then upload the approved
   PDF/HTML. The local recorded-evidence HTML is a preview of documented results,
   not proof of your upcoming fresh GPU execution.

The team notebook calls Qwen through local Transformers, so no paid LLM API key
or per-token provider balance is required. Colab GPU limits are separate.

### If an evaluation JSON response is malformed

The shared model boundary permits one schema-correction retry, preserving the
invalid raw output and error in the model-call log. No scoring defaults are
inserted. Two invalid responses still stop the run. After updating the code,
restart the Colab session or rerun the updated setup/import cells before model
execution. Save existing partial results first. A format retry is distinct from
an analysis revision and does not imply improved analysis quality.
