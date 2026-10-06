"""News ingest, preprocessing, FinBERT classification, extraction, and summary."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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

    def __init__(
        self,
        ticker_factory: Any | None = None,
        classifier: SentimentClassifier | None = None,
        news_tab: str = "news",
        cache_ttl_seconds: int = 900,
        fallback_news_sources: list[Any] | None = None,
        yahoo_timeout_seconds: int = 20,
    ) -> None:
        self._ticker_factory = ticker_factory
        self._classifier = classifier or FinBertClassifier()
        self._news_tab = news_tab
        self._cache_ttl_seconds = cache_ttl_seconds
        self._news_cache: dict[str, tuple[datetime, list[dict[str, Any]]]] = {}
        self._fallback_news_sources = fallback_news_sources
        self._yahoo_timeout_seconds = yahoo_timeout_seconds

    @staticmethod
    def _clean_text(value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text

    @staticmethod
    def _canonical_url(url: str) -> str:
        parts = urlsplit(url)
        # Drop tracking fields but retain the provider's actual article ID.
        # Finnhub serves individual articles from one path with `?id=...`.
        identity_query = urlencode([(key, value) for key, value in parse_qsl(parts.query) if key.lower() == "id"])
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), identity_query, ""))

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

    def _fetch_yahoo_news(self, ticker: str) -> list[dict[str, Any]]:
        """Get Yahoo search news with a bounded standard HTTP request."""
        if self._ticker_factory is not None:
            client = self._get_ticker(ticker)
            return list(client.get_news(count=10, tab=self._news_tab) if hasattr(client, "get_news") else client.news)

        try:
            import requests
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("Install project dependencies before using Yahoo Finance news.") from exc

        response = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": ticker, "newsCount": 10, "quotesCount": 1, "listsCount": 0},
            headers={"User-Agent": "Mozilla/5.0 (compatible; AAI520-research-agent/1.0)"},
            timeout=self._yahoo_timeout_seconds,
        )
        response.raise_for_status()
        return list(response.json().get("news", []))

    def ingest(self, ticker: str = "AAPL") -> tuple[list[dict[str, Any]], list[str]]:
        symbol = ticker.strip().upper()
        cached = self._news_cache.get(symbol)
        if cached and (datetime.now(UTC) - cached[0]).total_seconds() < self._cache_ttl_seconds:
            return cached[1], []
        yahoo_error: str | None = None
        try:
            # One standard Yahoo request per TTL window. A bounded GET avoids
            # yfinance's cookie/crumb POST path for news retrieval.
            raw_news = self._fetch_yahoo_news(symbol)
            if not raw_news:
                yahoo_error = "Yahoo Finance returned no news items."
            else:
                normalized = list(raw_news)
                self._news_cache[symbol] = (datetime.now(UTC), normalized)
                return normalized, []
        except Exception as exc:
            yahoo_error = f"Yahoo Finance news failed: {exc}"

        fallback_results = []
        for source in self._get_fallback_news_sources():
            result = source.news_feed(symbol)
            fallback_results.append(result)
            if result.ok and result.data.get("raw_news"):
                normalized = list(result.data["raw_news"])
                self._news_cache[symbol] = (datetime.now(UTC), normalized)
                return normalized, []

        fallback_errors = [error for result in fallback_results for error in result.errors]
        return [], [error for error in [yahoo_error, *fallback_errors] if error]

    def _get_fallback_news_sources(self) -> list[Any]:
        """Create independent live-news fallbacks only when they are needed."""
        if self._fallback_news_sources is not None:
            return self._fallback_news_sources
        from .alternative_sources import AlphaVantageClient, FinnhubClient

        # Finnhub is the preferred live source because Alpha Vantage's free
        # plan has a small daily request allowance.
        return [FinnhubClient(), AlphaVantageClient()]

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
