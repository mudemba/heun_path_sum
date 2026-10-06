# Computation of Heun functions via the Birkandan-Giscard-Tamar path-sum algorithm

`heun_path_sum` evaluates the local Heun function $H\ell(a, q; \alpha, \beta, \gamma, \delta; z)$ on a uniform grid of real or complex points, using only NumPy and SciPy. It implements the path-sum algorithm of T. Birkandan, P.-L. Giscard and A. Tamar, [Computations of general Heun functions from their integral series representations](https://ieeexplore.ieee.org/document/9598600) ([arXiv:2106.13729](https://arxiv.org/abs/2106.13729)).

The method is fast: tens of thousands of points in a fraction of a second. On a well-chosen grid the relative error is of order $10^{-7}$. The grid matters, so read [Choosing the grid](#choosing-the-grid) and [Accuracy and limits](#accuracy-and-limits) before relying on the numbers.

## Installation

```bash
git clone https://github.com/mudemba/heun_path_sum.git
cd heun_path_sum
pip install -r requirements.txt
```

The module is the single file `heun_path_sum.py`. It needs Python 3.9 or later, NumPy and SciPy 1.6 or later. Matplotlib is needed only to run `example.py`, and pytest only to run the tests.

## Usage

```python
import numpy as np
from heun_path_sum import heun

z = np.linspace(0.05, 0.9, 5000)
y = heun(z, a=4.3, q=-0.2, alpha=1.3, beta=0.12, gamma=1.6, delta=0.5)
```

`y` is a complex array holding $H\ell$ at each point of `z`. The six parameters are keyword-only and may be real or complex.

`python example.py` runs two worked examples and plots them: a check against the hypergeometric function, and a solution about $z = 1$. It prints

```text
Hypergeometric case: 20000 points in 0.165 s, largest relative error 9.0e-08
Solution about z = 1: 20000 points in 0.200 s
```

(times will differ on your machine).

## What is computed

Heun's general equation has regular singular points at $z = 0, 1, a$ and $\infty$:

$$
\frac{d^2y}{dz^2} + \left(\frac{\gamma}{z} + \frac{\delta}{z-1} + \frac{\epsilon}{z-a}\right)\frac{dy}{dz} + \frac{\alpha\beta z - q}{z(z-1)(z-a)}\,y = 0,
\qquad \epsilon = \alpha + \beta + 1 - \gamma - \delta .
$$

`heun` returns the local solution $H\ell$: the solution that is analytic at $z = 0$, normalised to $H\ell(0) = 1$. It exists when $a \neq 0$ and $\gamma$ is not zero or a negative integer.

| Argument | Meaning |
|---|---|
| `z_range` | Uniform grid of at least 3 points, real or complex. Its first point is where the solution is started. |
| `a` | Position of the third finite singular point. |
| `q` | Accessory parameter. |
| `alpha`, `beta`, `gamma`, `delta` | Exponent parameters; $\epsilon$ follows from them. |
| `max_sub_points` | Optional. The grid is solved in blocks of at most this many points (default 100). The default rarely needs changing. |
| `max_sub_width` | Optional. Largest distance spanned by one block (default 700). |

## Choosing the grid

The result is only as good as the grid, and three rules matter.

1. **Start inside the disc $|z| < \min(1, |a|)$, and not too close to zero.** The solution is started at the first point $z_0$ = `z_range[0]` from the power series of $H\ell$ about $z = 0$, summed until it stops changing. So $z_0$ may be anywhere inside the disc where that series converges, except very close to its edge, where it converges too slowly. The coefficients of the equation vary like $1/z$ near zero and the step has to resolve them, so use a step of a few per cent of $|z_0|$ or less. A first point further from zero therefore needs fewer points and gives a smaller error: for example, start at $0.05$ with a step of $5 \times 10^{-5}$. `heun` warns if the step is larger than $|z_0|$.
2. **Do not step across a singular point or a branch cut.** In general $H\ell$ has branch points at $z = 1$ and $z = a$, and a grid that runs through one of them returns numbers beyond it that are not values of $H\ell$. A grid containing $0$, $1$ or $a$ exactly is rejected. A complex grid may run along any straight line on which none of $z$, $z - 1$ and $a - z$ crosses the negative real axis; a grid that does cross is rejected, because the method's integrating factor is discontinuous there.
3. **For a solution about another singular point, change variable.** The solution of the same equation that is analytic at $z = 1$ is

   ```python
   heun(1 - z, a=1 - a, q=alpha*beta - q, alpha=alpha, beta=beta,
        gamma=delta, delta=gamma)
   ```

   with `1 - z` starting inside the corresponding disc, $|1 - z| < \min(1, |1 - a|)$. `example.py` does this for $\gamma = \delta = \epsilon = 1/2$, the algebraic form of Lamé's equation.

## Accuracy and limits

Every error figure in this section is recomputed by the [tests](#tests). The check used here is the case $\epsilon = 0$, $q = a\alpha\beta$, where $H\ell$ reduces exactly to the hypergeometric function ${}_2F_1(\alpha, \beta; \gamma; z)$. With $\alpha = 0.7$, $\beta = 1.1$, $\gamma = 1.6$ on `np.linspace(z0, 0.95, N)`, the largest relative error over the grid is:

| First point `z0` | N = 2,000 | N = 5,000 | N = 20,000 | N = 80,000 |
|---|---|---|---|---|
| 0.3 | 3.1e-07 | 7.3e-08 | 1.4e-08 | 3.4e-09 |
| 0.1 | 2.5e-06 | 4.4e-07 | 4.4e-08 | 6.9e-09 |
| 0.03 | 1.6e-05 | 2.3e-06 | 1.6e-07 | 1.5e-08 |
| 0.01 | 1.0e-04 | 9.9e-06 | 5.1e-07 | 3.7e-08 |
| 0.003 | 9.0e-04 | 7.1e-05 | 2.1e-06 | 1.2e-07 |
| 0.001 | 6.5e-03 | 5.3e-04 | 1.1e-05 | 3.8e-07 |

Evaluating 20,000 points takes about 0.15 s and 80,000 about 0.55 s on one 2.8 GHz core.

What the table shows:

- **The error keeps falling as the grid is refined.** The start values are accurate to rounding error, so there is no floor. Four times the points gives at least four times less error, and more while the start is still being resolved.
- **A first point close to zero is expensive.** At every grid size the top row is far more accurate than the bottom row, because the step has to resolve the start.

Other things to know:

- **An unresolved start is not recovered.** With a first point of $10^{-4}$ and 5,000 points, where the step is twice $|z_0|$, the error in this check is $2 \times 10^{-2}$. With a coarser start the result is meaningless.
- **A negative real part of $\gamma$ makes a close start worse.** For one parameter set with $\gamma = -0.4$ and 20,000 points, a first point of $0.001$ gave an error of $1.3 \times 10^{-3}$ and a first point of $0.05$ gave $1.2 \times 10^{-6}$.
- **Long intervals lose accuracy.** In the second example of `example.py`, at a fixed step of $10^{-4}$, the error at the end point is $4 \times 10^{-7}$ for an interval of length 2 and $1.5 \times 10^{-5}$ for one of length 20.

In a wider test against the power series, over 60 random real and complex parameter sets on grids from $0.05$ to $0.8$ along both real directions and along complex rays, the median error with 20,000 points was $8 \times 10^{-8}$ and the largest $2.8 \times 10^{-7}$. With 4,000 points the median was $1.6 \times 10^{-6}$ and the largest $5.6 \times 10^{-6}$. Some of these solutions pass through zero, where a relative error means little, so the error here is the largest absolute error divided by the largest value of $|H\ell|$ on the grid.

When more accuracy is needed than these figures, use the power series inside $|z| < \min(1, |a|)$, or an arbitrary-precision implementation such as Mathematica's `HeunG`.

## Tests

```bash
pytest
```

The tests check the start values against closed forms and a 30-digit evaluation, the handling of unusable input, and the accuracy figures above: every cell of the table and every error figure quoted in that section is recomputed and compared with the documented value. `pytest -s -k readme` prints each measured figure beside the documented one. The references are independent of the path-sum method: SciPy's hypergeometric function, a separately written power series, and direct integration of Heun's equation. Two of the tests use mpmath and are skipped if it is not installed. The whole suite takes about 20 seconds.

## How it works

Heun's equation is rewritten as a first-order system whose solution is a path-ordered exponential. Birkandan, Giscard and Tamar express that exponential through two scalar Volterra kernels $K_1$ and $K_2$ and their resolvents $G_i = K_i + K_i \ast K_i + \dots$, where $\ast$ is the Volterra composition:

$$
y(z) = y_0\left(1 + \int_{z_0}^{z} G_1\,d\zeta\right) + (y_0' - y_0)\left(e^{z - z_0} - 1 + \int_{z_0}^{z}\left(e^{z-\zeta} - 1\right)G_2(\zeta, z_0)\,d\zeta\right).
$$

The value $y_0$ and slope $y_0'$ at the first point come from the power series of $H\ell$ about $z = 0$. On a uniform grid, with the trapezoidal rule, each kernel becomes a triangular matrix and each resolvent a single triangular solve.

The grid is processed in consecutive blocks. Each block is started from the value and slope at the end of the one before, and that slope is the derivative of the same formula:

$$
y'(z) = y_0\,G_1(z, z_0) + (y_0' - y_0)\left(e^{z - z_0} + \int_{z_0}^{z} e^{z-\zeta}\,G_2(\zeta, z_0)\,d\zeta\right).
$$

## Differences from the reference implementation

The authors' own code is at [tbirkandan/integralseries_heun](https://github.com/tbirkandan/integralseries_heun). This implementation follows the same formulas and differs in these respects:

- The first sample of each resolvent is set to its exact value, $G(z_0, z_0) = K(z_0, z_0)$, where the reference sets it to zero. This removes an error proportional to the step from every block, which otherwise stops the result improving as the grid is refined.
- The start values come from the power series summed to convergence, at any first point inside its disc of convergence. The reference uses the first-order terms of the series at a fixed small $z_0$.
- The slope handed from one block to the next is the derivative of the path-sum formula at the block's last point. The reference uses a two-point backward difference of the computed values.
- Exponentials are only ever taken of differences within a block, never of $z$ itself, so grids at large $|z|$ do not overflow.
- The caller supplies any uniform grid, which is split automatically into blocks limited both in number of points and in width.

## Changes since version 1.0

Version 1.0 is the release archived on Zenodo. Since then:

- **The output has changed, and is more accurate**, for three reasons. Version 1.0 gave the first sample of each block's resolvent half its trapezoidal weight; it took the start values from the series to first order only; and it passed the slope between blocks as a three-point finite difference. In the check above its error levelled off near $3 \times 10^{-3}$ however fine the grid; it now falls as the table shows.
- The first point may be anywhere inside $|z| < \min(1, |a|)$ and no longer has to be close to zero. A first point outside that disc is rejected.
- Complex grids that cross a branch cut of the integrating factor are rejected. Version 1.0 returned wrong values beyond the crossing.
- `heun` warns when the step is larger than the distance of the first point from $z = 0$.
- Other unusable input (too few points, a non-uniform grid, a grid point on a singular point, $a = 0$, non-positive integer $\gamma$) raises a `ValueError` that says what is wrong.
- Parameters may be given as plain real numbers. Version 1.0 in general needed them to be complex-typed.
- Importing the module no longer requires Matplotlib, and `heun` no longer prints intermediate values.
- `max_sub_points` and `max_sub_width` can be set from `heun`.
- Evaluation is 1.5 to 2 times faster.
- Helper functions are private (their names start with an underscore). `heun` and `subdivide_domain` are the public interface.

## Citing

If you use this code, please cite the algorithm,

> T. Birkandan, P.-L. Giscard and A. Tamar, "Computations of general Heun functions from their integral series representations", *2021 Days on Diffraction (DD)*, IEEE, pp. 12-18 (2021). [arXiv:2106.13729](https://arxiv.org/abs/2106.13729)

and this implementation,

> M. Udemba, *heun_path_sum* (2026). Version 1.0: [doi:10.5281/zenodo.20732181](https://doi.org/10.5281/zenodo.20732181)

Version 1.0 is cited in P. Millington and M. Udemba, "Quantum corrections to symmetron fifth forces for planar sources", [arXiv:2606.28423](https://arxiv.org/abs/2606.28423).

## Licence

MIT. See [LICENSE](LICENSE).
