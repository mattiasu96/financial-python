from financial_python.simple_strategy import moving_average
import numpy as np
import cProfile
import pstats

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(1, 600):
        data = np.array([float(j) for j in range(1, i)], dtype=float)
        for j in range(1, i):
            ma = moving_average(data, window=j)
            # print(f"Window size {j}: {ma}")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats("moving_average_profile.prof")

