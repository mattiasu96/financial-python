import numpy as np

def simulate_price_path(num_days, num_assets, seed: int=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    # Parameters
    mu = 0.08      # expected annual return
    sigma = 0.20   # annual volatility
    dt = 1 / 252   # daily step

    # Daily log-returns
    daily_mu = mu * dt
    daily_sigma = sigma * np.sqrt(dt)
    
    # shape: (num_days - 1, num_assets)
    shocks = rng.normal(daily_mu, daily_sigma, size=(num_days - 1, num_assets))
    
    # log prices, start at 0, then cumsum shocks
    log_prices = np.zeros((num_days, num_assets))
    log_prices[1:] = np.cumsum(shocks, axis=0)
    
    # prices = exp(log_prices) * S0
    S0 = 100.0
    prices = S0 * np.exp(log_prices)
    return prices

def generate_random_weights(num_portfolios: int, num_assets: int, seed: int = 123) -> np.ndarray:
    """
    Generate num_portfolios random long-only portfolios (weights sum to 1).
    Returns:
        weights: array of shape (num_portfolios, num_assets)
    """
    rng = np.random.default_rng(seed)
    # Dirichlet gives non-negative weights that sum to 1
    weights = rng.dirichlet(alpha=np.ones(num_assets), size=num_portfolios)
    return weights

# ---------- STEP 3: Naive loop-based portfolio returns ----------

def compute_daily_asset_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute simple daily returns for each asset.
    prices: (T, N)
    returns: (T-1, N)
    """
    return prices[1:] / prices[:-1] - 1.0