# Multi-Agent Financial Analysis System

AAI-520 final team project by Tim Hays, Roopleen Kaur, and Ashley Marx.

The project will build a multi-agent workflow for a single-stock research run. The system will retrieve market and financial data plus news, plan and route research tasks, produce an evidence-based draft analysis, evaluate and revise the draft, and save reusable lessons for a later run.

## Team roles

| Team member | Primary responsibilities |
| --- | --- |
| Tim Hays | Team coordination, GitHub, data and news retrieval, prompt-chaining workflow |
| Roopleen Kaur | Research planning, routing, specialist integration, notebook assembly |
| Ashley Marx | Evaluation, revision loop, memory, final export and layout review |

## Planned workflow

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

## Setup

Create a virtual environment, install dependencies, then add required API keys to a local `.env` file. Do not commit API keys, large raw datasets, or generated notebook checkpoints.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project status

Week 1 focuses on the initial retrieval/news, planning/routing, and evaluation/memory prototypes. The team will agree on the model, tools, shared data format, and final source choices at kickoff.
