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
