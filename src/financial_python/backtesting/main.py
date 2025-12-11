from financial_python.backtesting.strategy import MovingAverageCrossoverStrategy
from financial_python.backtesting.data_loader import load_real_data
from financial_python.backtesting.plots import plot_moving_average_crossover_strategy

if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    data = load_real_data(tickers, start="2020-01-01", end="2023-01-01")

    # Initialize the strategy for one of the tickers, e.g., AAPL
    strategy = MovingAverageCrossoverStrategy(data['AAPL'], short_window=40, long_window=100)

    # Generate signals
    signals = strategy.generate_signals()

    print(signals.head())

    plot_moving_average_crossover_strategy(data['AAPL'], signals)

