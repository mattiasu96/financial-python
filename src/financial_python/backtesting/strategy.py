import pandas as pd


class TradingStrategy:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def generate_signals(self) -> pd.DataFrame:
        # Implement your signal generation logic here
        raise NotImplementedError("Should implement generate_signals()!")


class MovingAverageCrossoverStrategy(TradingStrategy):
    def __init__(
        self, data: pd.DataFrame, short_window: int = 40, long_window: int = 100
    ):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> pd.DataFrame:
        signals = pd.DataFrame(index=self.data.index)
        signals["signal"] = 0.0

        # Create short and long simple moving averages
        signals["short_mavg"] = self.data.rolling(
            window=self.short_window, min_periods=1
        ).mean()
        signals["long_mavg"] = self.data.rolling(
            window=self.long_window, min_periods=1
        ).mean()

        # TODO: fix improper indexing that causes SettingWithCopyWarning
        # Generate signals
        signals["signal"][self.short_window :] = (
            signals["short_mavg"][self.short_window :]
            > signals["long_mavg"][self.short_window :]
        ).astype(float)

        # TODO: maybe if instead of using diff i do the manual diff with shift it will be better?
        # Generate trading orders
        signals["positions"] = signals["signal"].diff()
        return signals
