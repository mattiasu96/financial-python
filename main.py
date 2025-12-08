from financial_python.simple_strategy import moving_average
import numpy as np


for i in range(1, 200):
    data = [float(j) for j in range(1, i)]
    for j in range(1, i):
        ma = moving_average(data, window=j)
        # print(f"Window size {j}: {ma}")

