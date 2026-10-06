from src.planning import ResearchPlanner


def test_route_selects_market_specialist():
    planner = ResearchPlanner()

    specialists = planner.route("What is Apple's stock performance?")

    assert specialists == ["market"]


def test_route_selects_multiple_specialists():
    planner = ResearchPlanner()

    specialists = planner.route(
        "What are Apple's recent earnings and news?"
    )

    assert specialists == ["earnings", "news"]


def test_normalize_ticker():
    planner = ResearchPlanner()

    assert planner.normalize_ticker(" aapl ") == "AAPL"


def test_empty_question_is_rejected():
    planner = ResearchPlanner()

    try:
        planner.route("")
    except ValueError as error:
        assert "research_question" in str(error)
    else:
        raise AssertionError("Empty research question should be rejected")

def test_plan_creates_research_steps():
    planner = ResearchPlanner()

    plan = planner.plan(
        "aapl",
        "What are Apple's recent earnings and news?"
    )

    assert plan.ticker == "AAPL"
    assert plan.research_question == "What are Apple's recent earnings and news?"
    assert plan.specialists == ["earnings", "news"]

    assert plan.steps == [
        "Retrieve evidence from the earnings specialist.",
        "Retrieve evidence from the news specialist.",
        "Combine the retrieved evidence into a draft analysis.",
        "Send the draft analysis and supporting evidence for evaluation.",
    ]


def test_plan_uses_company_fallback():
    planner = ResearchPlanner()

    plan = planner.plan(
        "aapl",
        "Tell me about Apple."
    )

    assert plan.specialists == ["company"]
    assert plan.steps[0] == (
        "Retrieve evidence from the company specialist."
    )

def test_execute_routes_earnings_to_financial_tool():
    class FakeTools:
        def financial_snapshot(self, ticker):
            return type(
                "FakeResult",
                (),
                {
                    "to_dict": lambda self: {
                        "tool": "financial_snapshot",
                        "ticker": ticker,
                        "data": {"revenue": 100},
                    }
                },
            )()

    planner = ResearchPlanner(research_tools=FakeTools())

    result = planner.execute(
        "aapl",
        "What are Apple's recent earnings?"
    )

    assert result["specialists"] == ["earnings"]
    assert result["evidence"]["earnings"]["data"]["revenue"] == 100

def test_execute_routes_news_to_news_pipeline():
    class FakeNewsPipeline:
        def process(self, ticker):
            return {
                "ticker": ticker,
                "items": [
                    {
                        "title": "Apple reports strong growth",
                        "sentiment": "positive",
                    }
                ],
            }

    planner = ResearchPlanner(news_pipeline=FakeNewsPipeline())

    result = planner.execute(
        "aapl",
        "What is the latest news about Apple?"
    )

    assert result["specialists"] == ["news"]
    assert result["evidence"]["news"]["ticker"] == "AAPL"
    assert result["evidence"]["news"]["items"][0]["sentiment"] == "positive"

def test_execute_routes_market_to_market_tool():
    class FakeTools:
        def market_snapshot(self, ticker):
            return type(
                "FakeResult",
                (),
                {
                    "to_dict": lambda self: {
                        "tool": "market_snapshot",
                        "ticker": ticker,
                        "data": {"latest_close": 250.0},
                    }
                },
            )()

    planner = ResearchPlanner(research_tools=FakeTools())

    result = planner.execute(
        "aapl",
        "What is Apple's stock price?"
    )

    assert result["specialists"] == ["market"]
    assert result["evidence"]["market"]["data"]["latest_close"] == 250.0


def test_execute_routes_company_to_profile_tool():
    class FakeTools:
        def company_profile(self, ticker):
            return type(
                "FakeResult",
                (),
                {
                    "to_dict": lambda self: {
                        "tool": "company_profile",
                        "ticker": ticker,
                        "data": {"sector": "Technology"},
                    }
                },
            )()

    planner = ResearchPlanner(research_tools=FakeTools())

    result = planner.execute(
        "aapl",
        "Tell me about Apple's company and sector."
    )

    assert result["specialists"] == ["company"]
    assert result["evidence"]["company"]["data"]["sector"] == "Technology"

def test_execute_market_with_real_research_tool():
    from src.data_tools import StockResearchTools

    planner = ResearchPlanner(
        research_tools=StockResearchTools()
    )

    result = planner.execute(
        "aapl",
        "What is Apple's stock performance?"
    )

    assert result["ticker"] == "AAPL"
    assert result["specialists"] == ["market"]
    assert "market" in result["evidence"]
    assert result["evidence"]["market"]["errors"] == []

def test_execute_earnings_with_real_research_tool():
    from src.data_tools import StockResearchTools

    planner = ResearchPlanner(
        research_tools=StockResearchTools()
    )

    result = planner.execute(
        "aapl",
        "What are Apple's recent earnings and financial results?"
    )

    assert result["ticker"] == "AAPL"
    assert result["specialists"] == ["earnings"]
    assert "earnings" in result["evidence"]
    assert result["evidence"]["earnings"]["errors"] == []

def test_execute_news_with_provided_news_evidence():
    class FakeNewsPipeline:
        def process(self, ticker):
            return {
                "ticker": ticker,
                "items": [
                    {
                        "title": "Apple reports strong revenue growth",
                        "sentiment": "positive",
                        "sentiment_score": 0.95,
                    }
                ],
                "errors": [],
            }

    planner = ResearchPlanner(news_pipeline=FakeNewsPipeline())

    result = planner.execute(
        "aapl",
        "What is Apple's latest news?"
    )

    assert result["specialists"] == ["news"]
    assert result["evidence"]["news"]["ticker"] == "AAPL"
    assert len(result["evidence"]["news"]["items"]) == 1
    assert result["evidence"]["news"]["items"][0]["sentiment"] == "positive"

def test_execute_combines_multiple_specialists():
    class FakeTools:
        def financial_snapshot(self, ticker):
            return type(
                "FakeResult",
                (),
                {
                    "to_dict": lambda self: {
                        "tool": "financial_snapshot",
                        "ticker": ticker,
                        "data": {"revenue": 100},
                    }
                },
            )()

    class FakeNewsPipeline:
        def process(self, ticker):
            return {
                "ticker": ticker,
                "items": [
                    {
                        "title": "Apple reports strong growth",
                        "sentiment": "positive",
                    }
                ],
                "errors": [],
            }

    planner = ResearchPlanner(
        research_tools=FakeTools(),
        news_pipeline=FakeNewsPipeline(),
    )

    result = planner.execute(
        "aapl",
        "What are Apple's recent earnings and news?"
    )

    assert result["specialists"] == ["earnings", "news"]

    assert "earnings" in result["evidence"]
    assert "news" in result["evidence"]

    assert result["evidence"]["earnings"]["data"]["revenue"] == 100
    assert result["evidence"]["news"]["items"][0]["sentiment"] == "positive"

    