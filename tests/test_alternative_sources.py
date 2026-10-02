from src.alternative_sources import AlphaVantageClient, FinnhubClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_alpha_vantage_normalizes_quote_and_news():
    def fake_get(url, params, timeout):
        if params["function"] == "GLOBAL_QUOTE":
            return FakeResponse({"Global Quote": {"05. price": "225.50", "08. previous close": "220.00", "10. change percent": "2.50%", "06. volume": "12345", "07. latest trading day": "2026-10-01"}})
        return FakeResponse({"feed": [{"title": "Apple earnings", "summary": "Revenue rose 8%.", "source": "Example Wire", "url": "https://example.com/apple", "time_published": "20261001T120000"}]})

    client = AlphaVantageClient(api_key="test", http_get=fake_get, min_interval_seconds=0)
    quote = client.market_snapshot("aapl")
    news = client.news_feed("AAPL")

    assert quote.ok and quote.data["latest_close"] == 225.5
    assert quote.data["change_pct"] == 2.5
    assert news.ok and news.data["raw_news"][0]["publisher"] == "Example Wire"


def test_finnhub_normalizes_quote_and_company_news():
    def fake_get(url, params, timeout):
        if url.endswith("/quote"):
            return FakeResponse({"c": 225.5, "pc": 220.0, "d": 5.5, "dp": 2.5, "h": 226.0, "l": 219.0, "t": 1790900000})
        return FakeResponse([{"headline": "Apple product news", "summary": "A product update was announced.", "source": "Example Wire", "url": "https://example.com/apple-news", "datetime": 1790900000}])

    client = FinnhubClient(api_key="test", http_get=fake_get)
    quote = client.market_snapshot("aapl")
    news = client.news_feed("AAPL")

    assert quote.ok and quote.data["latest_close"] == 225.5
    assert quote.data["as_of"].endswith("+00:00")
    assert news.ok and news.data["raw_news"][0]["title"] == "Apple product news"


def test_missing_provider_key_is_a_recoverable_tool_error():
    result = AlphaVantageClient(api_key="").market_snapshot("AAPL")

    assert not result.ok
    assert "ALPHA_VANTAGE_API_KEY" in result.errors[0]
