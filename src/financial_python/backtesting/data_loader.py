import yfinance as yf


def load_real_data(
    tickers: list[str], start: str = "2010-01-01", end: str = "2025-01-01"
):
    data = yf.download(tickers, start=start, end=end)[
        "Close"
    ]  # Can make the whole thing more complex by adding all the data
    return data.dropna()
