import numpy as np

POINTS = 5

X = np.linspace(1, 100, POINTS)

E = np.exp(X)

F = X**2

K_loop = np.zeros((POINTS, POINTS))
for i in range(POINTS):
    K_loop[i] = F*E/E[i]


fex, ey = np.meshgrid(F*E, 1/E, sparse=False)

K_meshgrid = fex*ey


# print(K_loop)
# print(K_meshgrid)
for i in range(POINTS):
    print(f"{K_loop[i]} vs {K_meshgrid[i]}")
