
import numpy as np
from numba import njit, jit
def portfolio_returns_loops(daily_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    VERY SLOW VERSION with nested loops.
    
    daily_returns: (T-1, N)
    weights: (P, N)
    Returns:
        port_returns: (P, T-1) -> each row is one portfolio's daily returns
    """
    num_days, num_assets = daily_returns.shape
    num_portfolios = weights.shape[0]
    
    port_returns = np.zeros((num_portfolios, num_days))
    
    for p in range(num_portfolios):            # loop over portfolios
        for t in range(num_days):             # loop over time
            r_t = 0.0
            for a in range(num_assets):       # loop over assets
                r_t += weights[p, a] * daily_returns[t, a]
            port_returns[p, t] = r_t
    
    return port_returns

#@jit(nopython=True)
def portfolio_returns_vectorized(daily_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Optimized version with vectorized operations.
    
    daily_returns: (T-1, N)
    weights: (P, N)
    Returns:
        port_returns: (P, T-1) -> each row is one portfolio's daily returns
    """
    return np.dot(weights, daily_returns.T)
