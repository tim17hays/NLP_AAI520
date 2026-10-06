"""Resilient, source-attributed market and company data retrieval tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable


@dataclass(frozen=True)
class SourceReference:
    """A source reference that downstream agents can retain in their reports."""

    name: str
    url: str
    accessed_at: str


@dataclass
class ToolResult:
    """A normalized tool response, including recoverable failure information."""

    tool: str
    ticker: str
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceReference] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ticker": self.ticker,
            "data": self.data,
            "sources": [asdict(source) for source in self.sources],
            "errors": self.errors,
        }


class StockResearchTools:
    """Retrieve AAPL (or another US ticker) evidence through ``yfinance``.

    The public methods intentionally return JSON-compatible ``ToolResult``
    values rather than yfinance/Pandas objects. This keeps the handoff stable
    for the planning, routing, and evaluation agents.
    """

    def __init__(self, ticker_factory: Callable[[str], Any] | None = None, request_timeout: int = 15) -> None:
        self._ticker_factory = ticker_factory
        self._request_timeout = request_timeout
        self._ticker_cache: dict[str, Any] = {}

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        normalized = ticker.strip().upper()
        if not normalized or not normalized.replace("-", "").isalnum():
            raise ValueError("ticker must contain letters/numbers and an optional hyphen")
        return normalized

    @staticmethod
    def _accessed_at() -> str:
        return datetime.now(UTC).isoformat()

    def _get_ticker(self, ticker: str) -> Any:
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]
        if self._ticker_factory:
            client = self._ticker_factory(ticker)
            self._ticker_cache[ticker] = client
            return client
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("Install project dependencies before using yfinance tools.") from exc
        client = yf.Ticker(ticker)
        self._ticker_cache[ticker] = client
        return client

    def company_profile(self, ticker: str = "AAPL") -> ToolResult:
        symbol = self.normalize_ticker(ticker)
        source = SourceReference("Yahoo Finance", f"https://finance.yahoo.com/quote/{symbol}", self._accessed_at())
        try:
            info = self._get_ticker(symbol).info or {}
            fields = (
                "longName", "sector", "industry", "longBusinessSummary", "marketCap",
                "currency", "website", "fullTimeEmployees",
            )
            return ToolResult("company_profile", symbol, {key: info.get(key) for key in fields if info.get(key) is not None}, [source])
        except Exception as exc:  # Network/provider failures must remain observable.
            return ToolResult("company_profile", symbol, sources=[source], errors=[str(exc)])

    def market_snapshot(self, ticker: str = "AAPL", period: str = "1mo") -> ToolResult:
        symbol = self.normalize_ticker(ticker)
        source = SourceReference("Yahoo Finance historical prices", f"https://finance.yahoo.com/quote/{symbol}/history", self._accessed_at())
        try:
            history = self._get_ticker(symbol).history(
                period=period,
                auto_adjust=False,
                timeout=self._request_timeout,
            )
            if history is None or history.empty:
                return ToolResult("market_snapshot", symbol, sources=[source], errors=["No historical price data returned."])
            closes = history["Close"].dropna()
            if closes.empty:
                return ToolResult("market_snapshot", symbol, sources=[source], errors=["Historical data had no closing prices."])
            first_close, latest_close = float(closes.iloc[0]), float(closes.iloc[-1])
            return ToolResult(
                "market_snapshot",
                symbol,
                {
                    "period": period,
                    "as_of": str(closes.index[-1]),
                    "latest_close": round(latest_close, 2),
                    "period_return_pct": round(((latest_close / first_close) - 1) * 100, 2),
                    "observations": int(len(closes)),
                },
                [source],
            )
        except Exception as exc:
            return ToolResult("market_snapshot", symbol, sources=[source], errors=[str(exc)])

    def financial_snapshot(self, ticker: str = "AAPL") -> ToolResult:
        symbol = self.normalize_ticker(ticker)
        source = SourceReference("Yahoo Finance financials", f"https://finance.yahoo.com/quote/{symbol}/financials", self._accessed_at())
        try:
            info = self._get_ticker(symbol).info or {}
            fields = ("totalRevenue", "grossProfits", "ebitda", "netIncomeToCommon", "trailingPE", "profitMargins")
            data = {key: info.get(key) for key in fields if info.get(key) is not None}
            if not data:
                return ToolResult("financial_snapshot", symbol, sources=[source], errors=["No financial fields returned."])
            return ToolResult("financial_snapshot", symbol, data, [source])
        except Exception as exc:
            return ToolResult("financial_snapshot", symbol, sources=[source], errors=[str(exc)])
