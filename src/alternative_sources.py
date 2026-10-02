"""Key-based live-data fallbacks for reliable Person 1 research retrieval.

The project begins with Yahoo Finance through ``yfinance``. These adapters add
two independent providers when Yahoo's news endpoint is empty or rate-limited:

* Alpha Vantage: quote, company overview, and market news/sentiment.
* Finnhub: quote, basic financial metrics, and company news.

Both return the same JSON-compatible ``ToolResult`` / raw-news shapes already
consumed by the project's planner and FinBERT news pipeline.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable

from .data_tools import SourceReference, StockResearchTools, ToolResult

HttpGet = Callable[..., Any]


class _KeyedClient:
    """Shared request, API-key, and provider-error handling."""

    provider_name = "External provider"
    env_key = ""

    def __init__(self, api_key: str | None = None, http_get: HttpGet | None = None) -> None:
        # Local `.env` stays untracked; explicit arguments still take priority.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:  # pragma: no cover - dependency-specific
            pass
        self.api_key = api_key or os.getenv(self.env_key, "")
        self._http_get = http_get

    def _get(self, url: str, *, params: dict[str, Any]) -> Any:
        if not self.api_key:
            raise RuntimeError(f"Set {self.env_key} in .env to use {self.provider_name}.")
        if self._http_get is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - dependency-specific
                raise RuntimeError("Install requests before using live-data fallbacks.") from exc
            http_get = requests.get
        else:
            http_get = self._http_get
        response = http_get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        return payload

    @staticmethod
    def _source(name: str, url: str) -> SourceReference:
        return SourceReference(name, url, datetime.now(UTC).isoformat())


class AlphaVantageClient(_KeyedClient):
    """Alpha Vantage fallback: quote, overview, and ``NEWS_SENTIMENT`` feed."""

    provider_name = "Alpha Vantage"
    env_key = "ALPHA_VANTAGE_API_KEY"
    base_url = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str | None = None,
        http_get: HttpGet | None = None,
        min_interval_seconds: float = 1.1,
    ) -> None:
        super().__init__(api_key=api_key, http_get=http_get)
        self._min_interval_seconds = min_interval_seconds
        self._last_request_at: float | None = None

    def _respect_request_interval(self) -> None:
        if self._last_request_at is None:
            self._last_request_at = time.monotonic()
            return
        remaining = self._min_interval_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _query(self, function: str, **params: Any) -> dict[str, Any]:
        # Free Alpha Vantage keys have a low burst limit. Pace requests rather
        # than retrying an immediate provider rejection.
        self._respect_request_interval()
        payload = self._get(self.base_url, params={"function": function, "apikey": self.api_key, **params})
        if not isinstance(payload, dict):
            raise RuntimeError("Alpha Vantage returned an unexpected response shape.")
        for error_key in ("Error Message", "Information", "Note"):
            if payload.get(error_key):
                raise RuntimeError(str(payload[error_key]))
        return payload

    def market_snapshot(self, ticker: str = "AAPL") -> ToolResult:
        symbol = StockResearchTools.normalize_ticker(ticker)
        source = self._source("Alpha Vantage GLOBAL_QUOTE", f"{self.base_url}?function=GLOBAL_QUOTE&symbol={symbol}")
        try:
            quote = self._query("GLOBAL_QUOTE", symbol=symbol).get("Global Quote", {})
            if not quote:
                raise RuntimeError("No quote fields returned.")
            return ToolResult(
                "market_snapshot",
                symbol,
                {
                    "as_of": quote.get("07. latest trading day"),
                    "latest_close": _float_or_none(quote.get("05. price")),
                    "previous_close": _float_or_none(quote.get("08. previous close")),
                    "change_pct": _percent_or_none(quote.get("10. change percent")),
                    "volume": _int_or_none(quote.get("06. volume")),
                },
                [source],
            )
        except Exception as exc:
            return ToolResult("market_snapshot", symbol, sources=[source], errors=[str(exc)])

    def financial_snapshot(self, ticker: str = "AAPL") -> ToolResult:
        symbol = StockResearchTools.normalize_ticker(ticker)
        source = self._source("Alpha Vantage OVERVIEW", f"{self.base_url}?function=OVERVIEW&symbol={symbol}")
        try:
            overview = self._query("OVERVIEW", symbol=symbol)
            fields = ("Name", "Sector", "Industry", "MarketCapitalization", "RevenueTTM", "EBITDA", "ProfitMargin", "PERatio")
            data = {field: overview[field] for field in fields if overview.get(field) not in (None, "", "None")}
            if not data:
                raise RuntimeError("No company-overview fields returned.")
            return ToolResult("financial_snapshot", symbol, data, [source])
        except Exception as exc:
            return ToolResult("financial_snapshot", symbol, sources=[source], errors=[str(exc)])

    def news_feed(self, ticker: str = "AAPL", limit: int = 10) -> ToolResult:
        symbol = StockResearchTools.normalize_ticker(ticker)
        source = self._source("Alpha Vantage NEWS_SENTIMENT", f"{self.base_url}?function=NEWS_SENTIMENT&tickers={symbol}")
        try:
            feed = self._query("NEWS_SENTIMENT", tickers=symbol, sort="LATEST", limit=limit).get("feed", [])
            records = [
                {
                    "title": item.get("title"),
                    "summary": item.get("summary"),
                    "publisher": item.get("source") or "Alpha Vantage",
                    "link": item.get("url"),
                    "published_at": item.get("time_published"),
                    "provider_sentiment": item.get("overall_sentiment_label"),
                }
                for item in feed
            ]
            if not records:
                raise RuntimeError("No news records returned.")
            return ToolResult("news_feed", symbol, {"raw_news": records}, [source])
        except Exception as exc:
            return ToolResult("news_feed", symbol, sources=[source], errors=[str(exc)])


class FinnhubClient(_KeyedClient):
    """Finnhub fallback: quote, basic financial metrics, and company news."""

    provider_name = "Finnhub"
    env_key = "FINNHUB_API_KEY"
    base_url = "https://finnhub.io/api/v1"

    def _query(self, path: str, **params: Any) -> Any:
        payload = self._get(f"{self.base_url}/{path.lstrip('/')}", params={"token": self.api_key, **params})
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return payload

    def market_snapshot(self, ticker: str = "AAPL") -> ToolResult:
        symbol = StockResearchTools.normalize_ticker(ticker)
        source = self._source("Finnhub quote", f"{self.base_url}/quote?symbol={symbol}")
        try:
            quote = self._query("quote", symbol=symbol)
            if quote.get("c") in (None, 0):
                raise RuntimeError("No current quote returned.")
            as_of = datetime.fromtimestamp(quote["t"], tz=UTC).isoformat() if quote.get("t") else None
            return ToolResult(
                "market_snapshot",
                symbol,
                {
                    "as_of": as_of,
                    "latest_close": quote.get("c"),
                    "previous_close": quote.get("pc"),
                    "change": quote.get("d"),
                    "change_pct": quote.get("dp"),
                    "day_high": quote.get("h"),
                    "day_low": quote.get("l"),
                },
                [source],
            )
        except Exception as exc:
            return ToolResult("market_snapshot", symbol, sources=[source], errors=[str(exc)])

    def financial_snapshot(self, ticker: str = "AAPL") -> ToolResult:
        symbol = StockResearchTools.normalize_ticker(ticker)
        source = self._source("Finnhub stock metric", f"{self.base_url}/stock/metric?symbol={symbol}&metric=all")
        try:
            metrics = self._query("stock/metric", symbol=symbol, metric="all").get("metric", {})
            fields = ("marketCapitalization", "revenuePerShareTTM", "epsTTM", "peTTM", "netProfitMarginTTM", "52WeekHigh", "52WeekLow")
            data = {field: metrics[field] for field in fields if metrics.get(field) is not None}
            if not data:
                raise RuntimeError("No basic financial metrics returned.")
            return ToolResult("financial_snapshot", symbol, data, [source])
        except Exception as exc:
            return ToolResult("financial_snapshot", symbol, sources=[source], errors=[str(exc)])

    def news_feed(self, ticker: str = "AAPL", days: int = 30) -> ToolResult:
        symbol = StockResearchTools.normalize_ticker(ticker)
        source = self._source("Finnhub company news", f"{self.base_url}/company-news?symbol={symbol}")
        end_date = date.today()
        try:
            payload = self._query("company-news", symbol=symbol, **{"from": (end_date - timedelta(days=days)).isoformat(), "to": end_date.isoformat()})
            if not isinstance(payload, list):
                raise RuntimeError("Company-news response was not a list.")
            records = [
                {
                    "title": item.get("headline"),
                    "summary": item.get("summary"),
                    "publisher": item.get("source") or "Finnhub",
                    "link": item.get("url"),
                    "published_at": item.get("datetime"),
                }
                for item in payload
            ]
            if not records:
                raise RuntimeError("No company-news records returned.")
            return ToolResult("news_feed", symbol, {"raw_news": records}, [source])
        except Exception as exc:
            return ToolResult("news_feed", symbol, sources=[source], errors=[str(exc)])


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value)) if value not in (None, "") else None
    except ValueError:
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(str(value)) if value not in (None, "") else None
    except ValueError:
        return None


def _percent_or_none(value: object) -> float | None:
    return _float_or_none(str(value).replace("%", ""))
