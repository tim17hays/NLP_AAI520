"""Execute the real five-stage news chain on labeled frozen classroom news."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import pipeline

    from src.news_pipeline import FinBertClassifier, NewsPipeline

    sample = json.loads(
        Path("data/examples/controlled_demo_inputs.json").read_text()
    )
    item = sample["SAMPLE_CONTEXT"]["evidence"]["news"]["items"][0]
    raw = [
        {
            "title": item["title"],
            "summary": item["summary"],
            "publisher": item["publisher"],
            "link": item["url"],
            "published_at": item["published_at"],
        }
    ]
    classifier = FinBertClassifier()
    classifier._pipeline = pipeline(
        "text-classification",
        model=args.model_path,
        tokenizer=args.model_path,
        framework="pt",
        device=-1,
    )
    result = NewsPipeline(classifier=classifier).process(
        "DEMO", raw_news=raw + raw
    )
    assert not result["errors"], result["errors"]
    assert len(result["items"]) == 1
    assert result["items"][0]["sentiment"] in {
        "positive",
        "neutral",
        "negative",
    }
    report = {
        "execution_mode": "fresh_local_finbert",
        "input_mode": "frozen fictional classroom news, duplicated to test preprocessing",
        "model": "ProsusAI/finbert",
        "model_snapshot": Path(args.model_path).name,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "raw_news": raw + raw,
        "result": result,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
