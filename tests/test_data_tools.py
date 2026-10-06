from src.data_tools import StockResearchTools


class FakeTicker:
    info = {"longName": "Apple Inc.", "sector": "Technology", "marketCap": 3000000000000}

    def history(self, period, auto_adjust, timeout):
        assert timeout == 15
        class Series:
            index = ["2026-10-01", "2026-10-02"]
            empty = False

            def dropna(self):
                return self

            def __len__(self):
                return 2

            class _ILoc:
                def __getitem__(self, value):
                    return [200.0, 210.0][value]

            iloc = _ILoc()

        class History:
            empty = False

            def __getitem__(self, key):
                assert key == "Close"
                return Series()

        return History()


def test_market_snapshot_has_normalized_handoff_payload():
    tools = StockResearchTools(ticker_factory=lambda _: FakeTicker())
    result = tools.market_snapshot("aapl")

    assert result.ok
    assert result.ticker == "AAPL"
    assert result.data["latest_close"] == 210.0
    assert result.data["period_return_pct"] == 5.0
    assert result.sources[0].url.endswith("/AAPL/history")


def test_invalid_ticker_is_rejected_before_lookup():
    tools = StockResearchTools(ticker_factory=lambda _: FakeTicker())
    try:
        tools.normalize_ticker("AAPL!")
    except ValueError as error:
        assert "ticker" in str(error)
    else:
        raise AssertionError("Invalid ticker should be rejected")


def test_ticker_client_is_reused_for_related_requests():
    calls = []

    def ticker_factory(symbol):
        calls.append(symbol)
        return FakeTicker()

    tools = StockResearchTools(ticker_factory=ticker_factory)
    tools.company_profile("AAPL")
    tools.market_snapshot("AAPL")

    assert calls == ["AAPL"]
