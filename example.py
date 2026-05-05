"""A script demonstrating proper use of the function path_sum() from the module heun"""

import time
import numpy as np
from matplotlib import pyplot as plt
import heun

# Heun parameters
A = -1.10193 * 1e8 + 1j*0
Q = 1.92837 * 1e6 + 1j*0
ALPHA = -1 + 1j*0
BETA = 3/2 + 1j*0
GAMMA = 0.5 + 1j*0
DELTA = 0.5 + 1j*0

PARAMS = [A, Q, ALPHA, BETA, GAMMA, DELTA]

# Boundary conditions
H0 = 0
H0_PRIME = 1

CONDITIONS = [H0, H0_PRIME]

# Domain definition
N = 1000
BUFFER = 1e-9

Z_MIN = 1 + BUFFER
# z_max = 8.64359*1e6
Z_MAX = 2
Z = np.linspace(Z_MIN, Z_MAX, N)

start = time.perf_counter()
y = heun.path_sum(CONDITIONS, PARAMS, 1 - Z)
end = time.perf_counter()

print(end - start)

plt.plot(Z, y)

plt.show()
