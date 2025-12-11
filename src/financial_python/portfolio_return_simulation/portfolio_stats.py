import numpy as np


def portfolio_stats_loops(port_returns: np.ndarray, trading_days: int = 252):
    """
    Compute cumulative return, annualized return, volatility, Sharpe per portfolio using loops.
    port_returns: (P, T-1)
    Returns:
        cumret, ann_ret, ann_vol, sharpe: each (P,)
    """
    num_portfolios, num_days = port_returns.shape

    cumret = np.zeros(num_portfolios)
    ann_ret = np.zeros(num_portfolios)
    ann_vol = np.zeros(num_portfolios)
    sharpe = np.zeros(num_portfolios)

    for p in range(num_portfolios):
        r = port_returns[p]  # daily returns for portfolio p

        growth = np.prod(1 + r)
        cumret[p] = growth - 1

        mean_daily = np.mean(r)
        std_daily = np.std(r, ddof=1)

        ann_ret[p] = (1 + mean_daily) ** trading_days - 1
        ann_vol[p] = std_daily * np.sqrt(trading_days)

        sharpe[p] = ann_ret[p] / ann_vol[p] if ann_vol[p] > 0 else np.nan

    return cumret, ann_ret, ann_vol, sharpe


def portfolio_stats_vectorized(port_returns: np.ndarray, trading_days: int = 252):
    cum_returns = np.prod(1 + port_returns, axis=1) - 1
    mean_daily_returns = np.mean(port_returns, axis=1)
    std_daily_returns = np.std(port_returns, axis=1, ddof=1)

    ann_returns = (1 + mean_daily_returns) ** trading_days - 1
    ann_volatilities = std_daily_returns * np.sqrt(trading_days)

    sharpe_ratios = np.where(
        ann_volatilities > 0, ann_returns / ann_volatilities, np.nan
    )

    return cum_returns, ann_returns, ann_volatilities, sharpe_ratios
