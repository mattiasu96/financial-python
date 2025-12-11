import matplotlib.pyplot as plt
import pandas as pd

def plot_moving_average_crossover_strategy(data: pd.DataFrame, signals: pd.DataFrame):
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
