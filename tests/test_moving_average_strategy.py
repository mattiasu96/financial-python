import pandas as pd
import numpy as np
from financial_python.backtesting.strategy import MovingAverageCrossoverStrategy
import matplotlib.pyplot as plt

def test_moving_average_strategy():

    # Create a sample DataFrame with synthetic data
    dates = pd.date_range('2023-01-01', periods=200)
    prices = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)
    data = pd.DataFrame(prices, columns=['Close'])

    # Initialize the strategy
    strategy = MovingAverageCrossoverStrategy(data['Close'], short_window=40, long_window=100)

    # Generate signals
    signals = strategy.generate_signals()

    # Basic assertions to check if signals are generated correctly
    assert 'signal' in signals.columns, "Signal column not found in the generated signals"
    assert 'positions' in signals.columns, "Positions column not found in the generated signals"
    assert len(signals) == len(data), "Signals DataFrame length does not match input data length"

    # Check that signals are either 0.0 or 1.0
    assert signals['signal'].isin([0.0, 1.0]).all(), "Signal values are not binary (0.0 or 1.0)"

    # Optional: Plot the results for visual inspection
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.plot(signals.index, signals['short_mavg'], label='40-day SMA')
    plt.plot(signals.index, signals['long_mavg'], label='100-day SMA')
    plt.plot(signals.loc[signals['positions'] == 1.0].index,
             data['Close'][signals['positions'] == 1.0],
             '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(signals.loc[signals['positions'] == -1.0].index,
             data['Close'][signals['positions'] == -1.0],
             'v', markersize=10, color='r', label='Sell Signal')
    plt.title('Moving Average Crossover Strategy Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


