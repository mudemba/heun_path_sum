"""A script demonstrating proper use of the function path_sum() from the module heun"""

import time
import numpy as np
from matplotlib import pyplot as plt
import heun_path_sum

# Heun parameters
A = 4.3 + 1j*0
Q = -0.2 + 1j*0
ALPHA = 1.3 + 1j*0
BETA = 0.12 + 1j*0
GAMMA = -0.4 + 1j*0
DELTA = 4.32 + 1j*0

PARAMS = [A, Q, ALPHA, BETA, GAMMA, DELTA]

# Domain definition
N = 1000
BUFFER = 1e-9

Z_MIN = -0.5
# z_max = 8.64359*1e6
Z_MAX = 0.5
Z = np.linspace(Z_MIN, Z_MAX, N)

start = time.perf_counter()
y = heun_path_sum.heun(Z, a=A, q=Q, alpha=ALPHA,
                       beta=BETA, gamma=GAMMA, delta=DELTA)
end = time.perf_counter()

print(end - start)

plt.plot(Z, y)

plt.xlabel("$z$")
plt.ylabel("$H\ell(z)$")

plt.show()
