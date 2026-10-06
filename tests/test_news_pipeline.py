from src.data_tools import SourceReference, ToolResult
from src.news_pipeline import NewsPipeline


class FakeFinBert:
    def classify(self, text):
        assert "Apple" in text
        return "positive", 0.9876


RAW_NEWS = [
    {
        "title": "Apple reports revenue growth",
        "summary": "Revenue increased 8% year over year.",
        "publisher": "Example Finance",
        "link": "https://example.com/apple-growth?tracking=1",
        "providerPublishTime": 1790900000,
    },
    {
        "title": "Duplicate Apple article",
        "summary": "Duplicate link should not be retained.",
        "publisher": "Example Finance",
        "link": "https://example.com/apple-growth",
        "providerPublishTime": 1790900000,
    },
]


def test_news_pipeline_deduplicates_classifies_and_preserves_evidence():
    result = NewsPipeline(classifier=FakeFinBert()).process("aapl", RAW_NEWS)

    assert result["ticker"] == "AAPL"
    assert result["workflow"] == ["ingest", "preprocess", "classify_finbert", "extract", "summarize"]
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["sentiment"] == "positive"
    assert item["sentiment_score"] == 0.9876
    assert item["source"]["url"] == "https://example.com/apple-growth"
    assert "Revenue increased 8% year over year." in item["extracted_facts"]


def test_yfinance_ingest_uses_one_cached_all_news_request():
    class FakeTicker:
        calls = 0

        def get_news(self, count, tab):
            self.calls += 1
            assert count == 10
            assert tab == "news"
            return RAW_NEWS

    ticker = FakeTicker()
    pipeline = NewsPipeline(ticker_factory=lambda _: ticker, classifier=FakeFinBert())

    first, first_errors = pipeline.ingest("AAPL")
    second, second_errors = pipeline.ingest("AAPL")

    assert not first_errors and not second_errors
    assert first == second == RAW_NEWS
    assert ticker.calls == 1


def test_news_pipeline_preserves_finnhub_article_ids_when_deduplicating():
    raw_news = [
        {"title": "Apple item one", "summary": "First report.", "publisher": "Finnhub", "link": "https://finnhub.io/api/news?id=one"},
        {"title": "Apple item two", "summary": "Second report.", "publisher": "Finnhub", "link": "https://finnhub.io/api/news?id=two"},
    ]
    result = NewsPipeline(classifier=FakeFinBert()).process("AAPL", raw_news)

    assert len(result["items"]) == 2
    assert result["items"][0]["source"]["url"].endswith("id=one")
    assert result["items"][1]["source"]["url"].endswith("id=two")


def test_news_pipeline_uses_live_provider_fallback_when_yahoo_is_empty():
    class EmptyYahooTicker:
        def get_news(self, count, tab):
            return []

    class FinnhubFallback:
        def news_feed(self, ticker):
            return ToolResult(
                "news_feed",
                ticker,
                {"raw_news": [RAW_NEWS[0]]},
                [SourceReference("Finnhub company news", "https://finnhub.io/api/v1/company-news?symbol=AAPL", "2026-10-05T00:00:00+00:00")],
            )

    pipeline = NewsPipeline(
        ticker_factory=lambda _: EmptyYahooTicker(),
        classifier=FakeFinBert(),
        fallback_news_sources=[FinnhubFallback()],
    )
    result = pipeline.process("AAPL")

    assert result["errors"] == []
    assert len(result["items"]) == 1
    assert result["items"][0]["publisher"] == "Example Finance"
