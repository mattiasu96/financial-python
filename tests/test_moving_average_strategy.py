import pandas as pd
import numpy as np
from financial_python.backtesting.strategy import MovingAverageCrossoverStrategy
import matplotlib.pyplot as plt
from financial_python.backtesting.plots import plot_moving_average_crossover_strategy


def test_moving_average_strategy():
    # Create a sample DataFrame with synthetic data
    dates = pd.date_range("2023-01-01", periods=200)
    prices = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)
    data = pd.DataFrame(prices, columns=["Close"])

    # Initialize the strategy
    strategy = MovingAverageCrossoverStrategy(
        data["Close"], short_window=40, long_window=100
    )

    # Generate signals
    signals = strategy.generate_signals()

    # Basic assertions to check if signals are generated correctly
    assert "signal" in signals.columns, (
        "Signal column not found in the generated signals"
    )
    assert "positions" in signals.columns, (
        "Positions column not found in the generated signals"
    )
    assert len(signals) == len(data), (
        "Signals DataFrame length does not match input data length"
    )

    # Check that signals are either 0.0 or 1.0
    assert signals["signal"].isin([0.0, 1.0]).all(), (
        "Signal values are not binary (0.0 or 1.0)"
    )
    # Optional: Plot the results for visual inspection
    plot_moving_average_crossover_strategy(data, signals, stock_name="Close")


def test_moving_average_strategy_real_data():
    from financial_python.backtesting.data_loader import load_real_data

    # Load real data for testing
    tickers = ["AAPL"]
    data = load_real_data(tickers=tickers, start="2020-01-01", end="2023-01-01")

    # Initialize the strategy
    strategy = MovingAverageCrossoverStrategy(
        data["AAPL"], short_window=40, long_window=100
    )

    # Generate signals
    signals = strategy.generate_signals()

    # Basic assertions to check if signals are generated correctly
    assert "signal" in signals.columns, (
        "Signal column not found in the generated signals"
    )
    assert "positions" in signals.columns, (
        "Positions column not found in the generated signals"
    )
    assert len(signals) == len(data), (
        "Signals DataFrame length does not match input data length"
    )

    # Check that signals are either 0.0 or 1.0
    assert signals["signal"].isin([0.0, 1.0]).all(), (
        "Signal values are not binary (0.0 or 1.0)"
    )

    # Optional: Plot the results for visual inspection
    plot_moving_average_crossover_strategy(data, signals, stock_name="AAPL")
