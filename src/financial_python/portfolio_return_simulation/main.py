import time

from financial_python.portfolio_return_simulation.simulate_price_path import (
    simulate_price_path,
    compute_daily_asset_returns,
    generate_random_weights,
)
from financial_python.portfolio_return_simulation.portfolio_return import (
    portfolio_returns_loops,
    portfolio_returns_vectorized,
)
from financial_python.portfolio_return_simulation.portfolio_stats import (
    portfolio_stats_loops,
    portfolio_stats_vectorized,
)

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    times_portfolios_loops = []
    times_portfolios_vectorized = []
    times_stats_loops = []
    times_stats_vectorized = []
    portfolio_sizes = list(range(1, 101))

    for P in portfolio_sizes:
        print(f"\n--- Testing with {P} portfolios ---")
        T, N = 2520, 50  # days, assets, portfolios

        prices = simulate_price_path(T, N)
        daily_returns = compute_daily_asset_returns(prices)
        weights = generate_random_weights(P, N)

        start = time.perf_counter()
        port_ret = portfolio_returns_loops(daily_returns, weights)
        end = time.perf_counter()
        print(f"Loop version took {end - start:.4f} seconds")
        times_portfolios_loops.append(end - start)

        start = time.perf_counter()
        port_ret_vec = portfolio_returns_vectorized(daily_returns, weights)
        end = time.perf_counter()
        print(f"Vectorized version took {end - start:.4f} seconds")
        times_portfolios_vectorized.append(end - start)

        start = time.perf_counter()
        stats = portfolio_stats_loops(port_ret)
        end = time.perf_counter()
        print(f"Loop version stats took {end - start:.4f} seconds")
        times_stats_loops.append(end - start)

        start = time.perf_counter()
        stats_vec = portfolio_stats_vectorized(port_ret_vec)
        end = time.perf_counter()
        print(f"Vectorized version stats took {end - start:.4f} seconds")
        times_stats_vectorized.append(end - start)

        # assert np.allclose(port_ret, port_ret_vec), "Results do not match!"

        # Plot grouped bar chart of timings vs portfolio size
    inds = np.arange(len(portfolio_sizes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        inds - 1.5 * width, times_portfolios_loops, width, label="Portfolios (loops)"
    )
    ax.bar(
        inds - 0.5 * width,
        times_portfolios_vectorized,
        width,
        label="Portfolios (vectorized)",
    )
    ax.bar(inds + 0.5 * width, times_stats_loops, width, label="Stats (loops)")
    ax.bar(
        inds + 1.5 * width, times_stats_vectorized, width, label="Stats (vectorized)"
    )

    # show fewer x-ticks for readability
    step = max(1, len(inds) // 10)
    ax.set_xticks(inds[::step])
    ax.set_xticklabels([str(s) for s in portfolio_sizes[::step]])
    ax.set_xlabel("Number of portfolios (P)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Timing comparison vs number of portfolios")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # save and show
    plt.savefig("timings_grouped_bar.png", dpi=150)
    plt.show()
