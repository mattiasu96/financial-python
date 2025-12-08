from typing import Sequence
import numpy as np

def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """
    Compute a simple moving average (SMA) of the input sequence.

    Parameters:
    - values: sequence of numeric values (list, tuple, or numpy array)
    - window: positive integer window size

    Returns:
    - numpy.ndarray of the same length as `values` where the first `window-1`
      entries are np.nan and the rest are the windowed averages.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")

    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return np.array([], dtype=float)

    # If window is 1 just return a copy
    if window == 1:
        return arr.copy()

    # cumsum trick to compute rolling sums efficiently
    cumsum = np.empty(n + 1, dtype=float)
    cumsum[0] = 0.0
    cumsum[1:] = np.cumsum(arr)

    result = np.full(n, np.nan, dtype=float)
    if window <= n:
        window_sums = cumsum[window:] - cumsum[:-window]
        result[window - 1 :] = window_sums / window

    return result