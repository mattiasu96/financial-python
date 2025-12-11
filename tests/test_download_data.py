from financial_python.backtesting.data_loader import load_real_data


def test_load_real_data():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    data = load_real_data(tickers)
    assert not data.empty, "DataFrame is empty"
    assert set(tickers).issubset(data.columns), (
        "Not all tickers are present in the DataFrame"
    )
