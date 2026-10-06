"""Two worked examples for heun_path_sum.heun().

1. A check against a closed form.  When epsilon = 0 and q = a*alpha*beta,
   the singular point at z = a drops out of Heun's equation and the local
   Heun function reduces to the Gauss hypergeometric function
   2F1(alpha, beta; gamma; z).

2. The solution that is analytic at z = 1, obtained from the same routine
   by the change of variable z -> 1 - z.  The parameters have
   gamma = delta = epsilon = 1/2, the algebraic form of Lame's equation.

Both grids start inside the disc where the power series about the
expansion point converges, with a step that is a small fraction of the
distance to that point, and neither steps across another singular point.
See "Choosing the grid" in the README.

Run with:  python example.py
"""

import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import hyp2f1  # pylint: disable=no-name-in-module

from heun_path_sum import heun

BLUE = "#2a78d6"
ORANGE = "#eb6834"


def hypergeometric_check():
    """Evaluate Hl where it equals 2F1; return the grid and both curves."""
    alpha, beta, gamma = 0.7, 1.1, 1.6
    a = 2.5
    z = np.linspace(0.05, 0.95, 20000)

    start = time.perf_counter()
    path_sum = heun(z, a=a, q=a*alpha*beta, alpha=alpha, beta=beta,
                    gamma=gamma, delta=alpha + beta + 1 - gamma)
    elapsed = time.perf_counter() - start

    exact = hyp2f1(alpha, beta, gamma, z)
    error = np.max(np.abs(path_sum - exact)/np.abs(exact))
    print(f"Hypergeometric case: {z.size} points in {elapsed:.3f} s, "
          f"largest relative error {error:.1e}")
    return z, path_sum, exact


def solution_about_one():
    """Evaluate the solution analytic at z = 1 on 1 < z <= 3."""
    a, q = -19.0954, 4.84052
    alpha, beta, gamma, delta = -1.0, 1.5, 0.5, 0.5
    z = np.linspace(1.05, 3, 20000)

    start = time.perf_counter()
    # About z = 1 the roles of gamma and delta swap, and
    # a -> 1 - a, q -> alpha*beta - q, z -> 1 - z.
    solution = heun(1 - z, a=1 - a, q=alpha*beta - q, alpha=alpha,
                    beta=beta, gamma=delta, delta=gamma)
    elapsed = time.perf_counter() - start

    print(f"Solution about z = 1: {z.size} points in {elapsed:.3f} s")
    return z, solution


def main():
    """Run both examples and plot them side by side."""
    z_left, path_sum, exact = hypergeometric_check()
    z_right, solution = solution_about_one()

    _, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    left.plot(z_left, path_sum.real, color=BLUE, linewidth=2,
              label="path sum")
    left.plot(z_left, exact, color=ORANGE, linewidth=2, linestyle="--",
              label="exact $_2F_1$")
    left.set_title("Hypergeometric case")
    left.set_ylabel("$Hl(z)$")
    left.legend(frameon=False)

    right.plot(z_right, solution.real, color=BLUE, linewidth=2)
    right.set_title("Solution analytic at $z = 1$")
    right.set_ylabel("$Hl(1 - z)$")

    for axes in (left, right):
        axes.set_xlabel("$z$")
        axes.grid(color="0.9", linewidth=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
