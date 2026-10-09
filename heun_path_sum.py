"""Local Heun functions by the Birkandan-Giscard-Tamar path-sum method.

Heun's general equation is the second-order linear equation with regular
singular points at z = 0, 1, a and infinity. It is given by

    y'' + (gamma/z + delta/(z - 1) + epsilon/(z - a)) y'
        + (alpha*beta*z - q) / (z*(z - 1)*(z - a)) y = 0,

where epsilon = alpha + beta + 1 - gamma - delta. The function heun()
evaluates the local solution Hl(a, q; alpha, beta, gamma, delta; z), the
solution that is analytic at z = 0 with Hl(0) = 1, on a uniform grid.

The method writes the solution as an integral series (a "path sum"),

    y(z) = y0 * (1 + int G1)
           + (y0' - y0) * (exp(z - z0) - 1
                           + int (exp(z - zeta) - 1) * G2(zeta, z0) dzeta),

where the integrals run from z0 to z, y0 and y0' are the value and slope
at the first grid point z0, and each G is the resolvent of a Volterra
kernel K, G = K + K*K + K*K*K + ... (* denotes Volterra compositions).
On a grid, the kernels become triangular matrices, and the resolvent is 
found via one triangular solve. The value and slope at z0 come from the 
power series of Hl about z = 0.

The grid is solved in consecutive blocks. Each block starts from the value
and slope at the end of the one before, and that slope is the derivative of
the same formula,

    y'(z) = y0 * G1(z, z0)
            + (y0' - y0) * (exp(z - z0)
                            + int exp(z - zeta) * G2(zeta, z0) dzeta).

Reference
---------
T. Birkandan, P.-L. Giscard and A. Tamar, "Computations of general Heun
functions from their integral series representations", 2021 Days on
Diffraction (DD), IEEE, pp. 12-18.  arXiv:2106.13729.

Example
-------
>>> import numpy as np
>>> from heun_path_sum import heun
>>> z = np.linspace(0.05, 0.9, 5000)
>>> y = heun(z, a=4.3, q=-0.2, alpha=1.3, beta=0.12, gamma=1.6, delta=0.5)
>>> y.shape, y.dtype
((5000,), dtype('complex128'))
"""

import sys
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_triangular

__all__ = ["heun", "subdivide_domain"]

# The grid is processed in blocks.  A block holds at most this many points,
# because the kernel matrices need memory quadratic in the block length.
DEFAULT_MAX_SUB_POINTS = 100

# A block spans at most this distance, so that exp(z - z0) stays below the
# largest double (exp(709.78...)) however long the whole grid is.
DEFAULT_MAX_SUB_WIDTH = 700.0

# Most terms summed for the start values.  The terms shrink like
# (abs(z0)/R)**n, where R = min(1, abs(a)) is the radius of convergence, so
# the limit is reached only when the first point is within about half a
# per cent of the edge of the disc.
SERIES_MAX_TERMS = 10000


def _heun_eq_coeff_1(z_range: np.ndarray, a: complex, gamma: complex,
                     delta: complex, epsilon: complex) -> np.ndarray:
    """Coefficient of y' in Heun's equation."""
    return gamma/z_range + delta/(z_range - 1) + epsilon/(z_range - a)


def _heun_eq_coeff_0(z_range: np.ndarray, a: complex, q: complex,
                     alpha: complex, beta: complex) -> np.ndarray:
    """Coefficient of y in Heun's equation."""
    return (alpha * beta * z_range - q) / (
        z_range * (z_range - 1) * (z_range - a))


def _weight_func(z_range: np.ndarray, a: complex, gamma: complex,
                 delta: complex, epsilon: complex) -> np.ndarray:
    """Integrating factor z^gamma (z - 1)^delta (a - z)^epsilon.

    Principal branches are used. Only ratios of this function at two
    points of the same block enter the result.
    """
    return (z_range**gamma) * ((z_range - 1)**delta) * ((a - z_range)**epsilon)


def _kernel_1(z_range: np.ndarray, x_vec: np.ndarray, y_vec: np.ndarray,
              delta_z: complex) -> np.ndarray:
    """Matrix of the first kernel.

    Entry [i, j] holds, for j >= i,

        K1(z_j, z_i) = 1 + (1/w(z_j)) * int_{z_i}^{z_j} X(s) w(s) ds,

    with w(s) = y(s) exp(s), y the integrating factor and X = -P - Q - 1,
    where P and Q are the coefficients of y' and y in Heun's equation.
    The exponential is taken relative to the first point of the block,
    which leaves the ratio unchanged and cannot overflow. Entries below
    the diagonal are not used.
    """
    weight = y_vec*np.exp(z_range - z_range[0])
    integral = cumulative_trapezoid(x_vec*weight, axis=0, initial=0)
    differences = integral[np.newaxis, :] - integral[:, np.newaxis]
    return 1 + delta_z*differences/weight[np.newaxis, :]


def _kernel_2(z_range: np.ndarray, x_vec: np.ndarray,
              q_vec: np.ndarray) -> np.ndarray:
    """Matrix of the second kernel.

    Entry [i, j] holds, for j >= i,

        K2(z_j, z_i) = X(z_j) exp(z_j - z_i) + Q(z_j),

    where Q is the coefficient of y in Heun's equation. Entries below the
    diagonal are not used.
    """
    growth = np.exp(z_range[np.newaxis, :] - z_range[:, np.newaxis])
    return x_vec[np.newaxis, :]*growth + q_vec[np.newaxis, :]


def _neumann_sum(kernel: np.ndarray, delta_z: complex) -> np.ndarray:
    """Resolvent of a Volterra kernel along the block, from one solve.

    Returns delta_z * G(z_j, z_0) for every point z_j of the block, where
    G = K + K*K + K*K*K + ... solves G = K + K*G. With the compositions
    discretised by the trapezoidal rule, the Neumann series becomes the
    triangular system solved here.

    The solve gives the first sample only half its weight, because the
    trapezoidal rule halves the diagonal. It is therefore set directly:
    at coincident points the composition integral is empty, so
    G(z_0, z_0) = K(z_0, z_0) exactly.
    """
    points = kernel.shape[0]
    diagonal = np.diag(np.diag(kernel))
    identity = np.identity(points, dtype=complex)
    source = identity[0]

    lhs = identity - delta_z*kernel + 0.5*delta_z*diagonal

    green = solve_triangular(lhs, source, trans=1) - source
    green[0] = delta_z*kernel[0, 0]
    return green


def _path_ordered_exp_1(z_range: np.ndarray, x_vec: np.ndarray,
                        y_vec: np.ndarray, delta_z: complex
                        ) -> tuple[np.ndarray, complex]:
    """Factor multiplying y0 in the path sum, and its slope at the end.

    Returns 1 + int G1 at every point of the block, and its derivative
    G1(z, z0) at the last point.
    """
    green = _neumann_sum(_kernel_1(z_range, x_vec, y_vec, delta_z), delta_z)
    factor = 1 + cumulative_trapezoid(green, axis=0, initial=0)
    return factor, green[-1]/delta_z


def _path_ordered_exp_2(z_range: np.ndarray, x_vec: np.ndarray,
                        q_vec: np.ndarray, delta_z: complex
                        ) -> tuple[np.ndarray, complex]:
    """Factor multiplying (y0' - y0) in the path sum, and its end slope.

    Returns exp(z - z0) - 1 + int (exp(z - zeta) - 1) G2(zeta, z0) dzeta
    at every point of the block, and its derivative
    exp(z - z0) + int exp(z - zeta) G2(zeta, z0) dzeta at the last point.
    Every exponential is taken of a difference bounded by the block width.
    """
    green = _neumann_sum(_kernel_2(z_range, x_vec, q_vec), delta_z)
    growth = np.exp(z_range - z_range[0])

    exp_green = np.exp(-z_range + z_range[0]) * green
    int_part_1 = growth * cumulative_trapezoid(exp_green, axis=0, initial=0)
    int_part_2 = cumulative_trapezoid(green, axis=0, initial=0)

    factor = growth - 1 + int_part_1 - int_part_2
    return factor, growth[-1] + int_part_1[-1]


def _start_values(z0: complex, a: complex, q: complex, alpha: complex,
                  beta: complex, gamma: complex,
                  delta: complex) -> tuple[complex, complex]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    """Value and slope of Hl at z0, from its power series about z = 0.

    Hl(z) = sum_n c_n z^n with c_0 = 1, c_1 = q/(a gamma) and

        a (n + 1)(n + gamma) c_{n+1} = (Q_n + q) c_n - P_n c_{n-1},
        Q_n = n ((n - 1 + gamma)(1 + a) + a delta + epsilon),
        P_n = (n - 1 + alpha)(n - 1 + beta),

    which converges for abs(z) < min(1, abs(a)). The recurrence is run on
    the terms t_n = c_n z0^n rather than on the coefficients, which can
    overflow, and stops once two terms in a row are too small to change
    either sum.
    """
    z0 = complex(z0)
    epsilon = 1 + alpha + beta - gamma - delta
    rounding = sys.float_info.epsilon

    previous, term = 1 + 0j, q*z0/(a*gamma)     # t_0 and t_1
    value, weighted = previous + term, term     # sum of t_n, sum of n t_n
    largest, largest_weighted = max(1.0, abs(term)), abs(term)
    negligible_in_a_row = 0

    for n in range(1, SERIES_MAX_TERMS):
        p_n = (n - 1 + alpha)*(n - 1 + beta)
        q_n = n*((n - 1 + gamma)*(1 + a) + a*delta + epsilon)
        r_n = a*(n + 1)*(n + gamma)
        previous, term = term, z0*((q_n + q)*term - p_n*z0*previous)/r_n

        value += term
        weighted += (n + 1)*term
        largest = max(largest, abs(term))
        largest_weighted = max(largest_weighted, (n + 1)*abs(term))

        if (abs(term) <= rounding*largest
                and (n + 1)*abs(term) <= rounding*largest_weighted):
            negligible_in_a_row += 1
            if negligible_in_a_row == 2:
                return value, weighted/z0
        else:
            negligible_in_a_row = 0

    raise ValueError(
        "the power series for the start values did not converge: move "
        "z_range[0] further inside abs(z) < min(1, abs(a))")


def _is_nonpositive_integer(value: complex) -> bool:
    """True for 0, -1, -2, ... (with zero imaginary part)."""
    return value.imag == 0 and value.real <= 0 and value.real.is_integer()


def _crosses_branch_cut(z_range: np.ndarray, a: complex) -> bool:
    """True if the integrating factor jumps somewhere along the grid.

    The factor z^gamma (z - 1)^delta (a - z)^epsilon is evaluated on
    principal branches, so it is discontinuous wherever z, z - 1 or a - z
    crosses the negative real axis. That can only happen on a grid that
    is not real.
    """
    for factor in (z_range, z_range - 1, a - z_range):
        phase = np.angle(factor)
        if np.any(np.unwrap(phase) != phase):
            return True
    return False


def subdivide_domain(domain: np.ndarray,
                     max_sub_points: int = DEFAULT_MAX_SUB_POINTS,
                     max_sub_width: float = DEFAULT_MAX_SUB_WIDTH
                     ) -> list[np.ndarray]:
    """Split a grid into consecutive blocks that share their end points.

    Each block holds at most max_sub_points points and spans at most
    max_sub_width. The last point of a block is the first point of the
    next, so the blocks can be solved one after another.

    Parameters
    ----------
    domain : np.ndarray
        One-dimensional grid.
    max_sub_points : int, optional
        Largest number of points in a block.  Must be at least 2.
    max_sub_width : float, optional
        Largest distance between the first and last point of a block.

    Returns
    -------
    list of np.ndarray
        Views into domain, in order. Empty if domain is empty.
    """
    if max_sub_points < 2:
        raise ValueError("max_sub_points must be at least 2")

    points = len(domain)
    if points == 0:
        return []

    subdomains = []
    start_index = 0

    for i in range(1, points):
        would_exceed_points = (i + 1 - start_index) > max_sub_points
        would_exceed_width = np.abs(
            domain[i] - domain[start_index]) > max_sub_width

        if would_exceed_points or would_exceed_width:
            subdomains.append(domain[start_index:i])
            start_index = i - 1

    subdomains.append(domain[start_index:])
    return subdomains


def _validated_grid(z_range, a: complex, max_sub_points: int,
                    max_sub_width: float) -> np.ndarray:
    """Return z_range as an array, or raise ValueError if it is unusable."""
    z_range = np.asarray(z_range)
    if not np.issubdtype(z_range.dtype, np.inexact):
        z_range = z_range.astype(float)

    if z_range.ndim != 1 or z_range.size < 3:
        raise ValueError(
            "z_range must be a one-dimensional grid of at least 3 points")
    if not np.all(np.isfinite(z_range)):
        raise ValueError("z_range must be finite")

    delta_z = z_range[1] - z_range[0]
    if delta_z == 0:
        raise ValueError("z_range must have a non-zero step")
    # np.linspace rounds each point, so allow for that as well as for a
    # small relative error in the step.
    tolerance = 1e-6*abs(delta_z) + 1e-12*np.max(np.abs(z_range))
    if np.max(np.abs(np.diff(z_range) - delta_z)) > tolerance:
        raise ValueError("z_range must be uniformly spaced")

    if np.any((z_range == 0) | (z_range == 1) | (z_range == a)):
        raise ValueError(
            "z_range must not contain the singular points 0, 1 or a")
    if abs(z_range[0]) >= min(1, abs(a)):
        raise ValueError(
            "z_range[0] must lie inside abs(z) < min(1, abs(a)), where the "
            "power series that starts the solution converges")
    if _crosses_branch_cut(z_range, a):
        raise ValueError(
            "along z_range, one of z, z - 1 and a - z crosses the negative "
            "real axis, where the integrating factor is discontinuous")

    if max_sub_points < 3:
        raise ValueError("max_sub_points must be at least 3")
    if max_sub_width <= 0 or 2*abs(delta_z) > max_sub_width:
        raise ValueError(
            "max_sub_width must be at least twice the grid step")

    return z_range


def heun(z_range: np.ndarray, *, a: complex, q: complex,
         alpha: complex, beta: complex, gamma: complex, delta: complex,
         max_sub_points: int = DEFAULT_MAX_SUB_POINTS,
         max_sub_width: float = DEFAULT_MAX_SUB_WIDTH) -> np.ndarray:
    # pylint: disable=too-many-arguments,too-many-locals
    """Evaluate the local Heun function Hl on a uniform grid.

    Hl(a, q; alpha, beta, gamma, delta; z) is the solution of Heun's
    general equation that is analytic at z = 0 and equals 1 there.

    Parameters
    ----------
    z_range : np.ndarray
        Uniformly spaced grid of at least 3 points, real or complex. The
        solution is started at z_range[0] from the power series about
        z = 0, so the first point must lie inside the disc where that
        series converges, abs(z) < min(1, abs(a)), and not at 0.  The grid
        must not contain a singular point (0, 1 or a), and should not step
        across one.  A complex grid may run in any direction along which
        none of z, z - 1 and a - z crosses the negative real axis.
    a : complex
        Position of the third finite singular point. Must not be 0.
    q : complex
        Accessory parameter.
    alpha, beta, gamma, delta : complex
        Exponent parameters.  The fifth, epsilon, is fixed by
        epsilon = alpha + beta + 1 - gamma - delta. gamma must not be
        zero or a negative integer, where Hl is not defined.
    max_sub_points : int, optional
        The grid is solved in consecutive blocks of at most this many
        points. The default rarely needs changing; memory grows with the
        square of the block length.  Must be at least 3.
    max_sub_width : float, optional
        Largest distance spanned by a block.  The default keeps
        exp(z - z0) within double precision.

    Returns
    -------
    np.ndarray
        Complex array of Hl at each point of z_range.

    Raises
    ------
    ValueError
        If the grid or the parameters are outside the conditions above.

    Warns
    -----
    RuntimeWarning
        If the step is larger than abs(z_range[0]).

    Notes
    -----
    The start values are accurate to rounding, so the error comes from
    the discretisation and falls as the grid is refined, a little faster
    than in proportion to the step. The step has to resolve the
    coefficients of the equation, which vary like 1/z near z = 0: a step 
    of a few per cent of abs(z_range[0]) or less is a good choice, and a
    step larger than abs(z_range[0]) gives a start that no later
    refinement recovers. Starting further from z = 0, where a given step
    resolves more, is therefore cheaper than starting close.

    The local solution about another singular point follows from the same
    routine by a change of variable.  About z = 1, for example,

        heun(1 - z, a=1 - a, q=alpha*beta - q, alpha=alpha, beta=beta,
             gamma=delta, delta=gamma)

    is the solution of the original equation that is analytic at z = 1.
    """
    a, q, alpha, beta, gamma, delta = (
        complex(value) for value in (a, q, alpha, beta, gamma, delta))

    if a == 0:
        raise ValueError("a must not be 0")
    if _is_nonpositive_integer(gamma):
        raise ValueError("gamma must not be zero or a negative integer")

    z_range = _validated_grid(z_range, a, max_sub_points, max_sub_width)
    epsilon = 1 + alpha + beta - gamma - delta
    delta_z = z_range[1] - z_range[0]

    if abs(delta_z) > abs(z_range[0]):
        warnings.warn(
            "the grid step is larger than the distance of the first point "
            "from z = 0, so the start is not resolved and the result will "
            "be inaccurate", RuntimeWarning, stacklevel=2)

    init_val, init_slope = _start_values(
        z_range[0], a, q, alpha, beta, gamma, delta)

    heun_function = np.empty(z_range.size, dtype=complex)
    filled = 0

    for subinterval in subdivide_domain(z_range, max_sub_points,
                                        max_sub_width):
        p_func = _heun_eq_coeff_1(subinterval, a, gamma, delta, epsilon)
        q_func = _heun_eq_coeff_0(subinterval, a, q, alpha, beta)
        x_func = - p_func - q_func - 1
        y_func = _weight_func(subinterval, a, gamma, delta, epsilon)

        factor_1, end_slope_1 = _path_ordered_exp_1(
            subinterval, x_func, y_func, delta_z)
        factor_2, end_slope_2 = _path_ordered_exp_2(
            subinterval, x_func, q_func, delta_z)

        contribution = init_val*factor_1
        contribution += (init_slope - init_val)*factor_2
        end_slope = (init_val*end_slope_1
                     + (init_slope - init_val)*end_slope_2)

        # Consecutive blocks share an end point; keep the earlier value.
        new_values = contribution if filled == 0 else contribution[1:]
        heun_function[filled:filled + new_values.size] = new_values
        filled += new_values.size

        # The next block starts from the value and slope at that point.
        init_val = heun_function[filled - 1]
        init_slope = end_slope

    return heun_function
