"""News ingest, preprocessing, FinBERT classification, extraction, and summary."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

from .data_tools import SourceReference


class SentimentClassifier(Protocol):
    """Small interface that makes classification independently testable."""

    def classify(self, text: str) -> tuple[str, float]: ...


@dataclass(frozen=True)
class NewsItem:
    ticker: str
    title: str
    summary: str
    publisher: str
    url: str
    published_at: str | None
    sentiment: str
    sentiment_score: float
    extracted_facts: list[str]
    source: SourceReference

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["source"] = asdict(self.source)
        return output


class FinBertClassifier:
    """Lazy Hugging Face adapter for ProsusAI/finbert financial sentiment."""

    model_name = "ProsusAI/finbert"

    def __init__(self) -> None:
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - dependency-specific
                raise RuntimeError("Install transformers and torch to run FinBERT classification.") from exc
            self._pipeline = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name)
        return self._pipeline

    def classify(self, text: str) -> tuple[str, float]:
        result = self._load()(text, truncation=True, max_length=512)
        prediction = result[0] if isinstance(result, list) else result
        return str(prediction["label"]).lower(), round(float(prediction["score"]), 4)


class NewsPipeline:
    """Complete required news prompt-chain preparation with traceable evidence.

    The final summarization here is intentionally extractive: it preserves the
    source wording and citation for Person 2's drafting agent rather than
    making an unsupported market claim before the specialist reviews it.
    """

    def __init__(self, ticker_factory: Any | None = None, classifier: SentimentClassifier | None = None) -> None:
        self._ticker_factory = ticker_factory
        self._classifier = classifier or FinBertClassifier()

    @staticmethod
    def _clean_text(value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text

    @staticmethod
    def _canonical_url(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))

    @staticmethod
    def _published_at(raw: object) -> str | None:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw, tz=UTC).isoformat()
        return str(raw) if raw else None

    @staticmethod
    def _extract_facts(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        facts = [sentence.strip() for sentence in sentences if re.search(r"\d|%|\$|revenue|earnings|guidance", sentence, re.I)]
        return facts[:3] or ([sentences[0].strip()] if sentences and sentences[0].strip() else [])

    def _get_ticker(self, ticker: str) -> Any:
        if self._ticker_factory:
            return self._ticker_factory(ticker)
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("Install project dependencies before using yfinance news.") from exc
        return yf.Ticker(ticker)

    @staticmethod
    def _yahoo_rss_news(ticker: str) -> list[dict[str, Any]]:
        """Use Yahoo Finance's public RSS as a no-key fallback for yfinance."""
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency-specific
            raise RuntimeError("Install requests before using the Yahoo Finance RSS fallback.") from exc
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        records: list[dict[str, Any]] = []
        for item in root.findall("./channel/item"):
            records.append(
                {
                    "title": item.findtext("title"),
                    "summary": item.findtext("description"),
                    "publisher": "Yahoo Finance RSS",
                    "link": item.findtext("link"),
                    "published_at": item.findtext("pubDate"),
                }
            )
        return records

    def ingest(self, ticker: str = "AAPL") -> tuple[list[dict[str, Any]], list[str]]:
        symbol = ticker.strip().upper()
        try:
            ticker_client = self._get_ticker(symbol)
            # ``Ticker.news`` can be empty with newer yfinance releases even
            # when the query-backed ``get_news`` endpoint has results.
            raw_news = ticker_client.news or []
            if not raw_news and hasattr(ticker_client, "get_news"):
                raw_news = ticker_client.get_news(count=10) or []
            if not raw_news:
                raw_news = self._yahoo_rss_news(symbol)
            if not raw_news:
                return [], ["No Yahoo Finance news items returned."]
            return list(raw_news), []
        except Exception as exc:
            return [], [str(exc)]

    def process(self, ticker: str = "AAPL", raw_news: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Return a stable handoff payload for the planner/routing agent."""
        symbol = ticker.strip().upper()
        errors: list[str] = []
        if raw_news is None:
            raw_news, errors = self.ingest(symbol)
        seen_urls: set[str] = set()
        items: list[NewsItem] = []
        for entry in raw_news:
            content = entry.get("content", entry)
            title = self._clean_text(content.get("title") or entry.get("title"))
            summary = self._clean_text(content.get("summary") or content.get("description") or entry.get("summary"))
            url = self._clean_text(content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else content.get("link") or entry.get("link"))
            canonical_url = self._canonical_url(url) if url else ""
            if not title or not canonical_url or canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)
            publisher = self._clean_text(content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else entry.get("publisher")) or "Unknown publisher"
            published = self._published_at(content.get("pubDate") or entry.get("providerPublishTime") or entry.get("published_at"))
            text = self._clean_text(f"{title}. {summary}")
            try:
                label, score = self._classifier.classify(text)
            except Exception as exc:
                errors.append(f"FinBERT classification failed for '{title}': {exc}")
                label, score = "unavailable", 0.0
            source = SourceReference(publisher, canonical_url, datetime.now(UTC).isoformat())
            items.append(NewsItem(symbol, title, summary, publisher, canonical_url, published, label, score, self._extract_facts(text), source))

        return {
            "ticker": symbol,
            "workflow": ["ingest", "preprocess", "classify_finbert", "extract", "summarize"],
            "items": [item.to_dict() for item in items],
            "summary": self._summarize(items),
            "errors": errors,
        }

    @staticmethod
    def _summarize(items: list[NewsItem]) -> str:
        if not items:
            return "No deduplicated news evidence was available for this run."
        labels: dict[str, int] = {}
        for item in items:
            labels[item.sentiment] = labels.get(item.sentiment, 0) + 1
        sentiment_counts = ", ".join(f"{count} {label}" for label, count in sorted(labels.items()))
        lead_titles = "; ".join(item.title for item in items[:3])
        return f"Processed {len(items)} source-attributed news items ({sentiment_counts}). Representative headlines: {lead_titles}."
