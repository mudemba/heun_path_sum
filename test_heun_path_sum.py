"""Tests for heun_path_sum.

Run with:  pytest
Add -s to print each measured error beside the figure quoted in the README.

The accuracy tests reproduce the numbers in the README's "Accuracy and
limits" section, using three references that do not depend on the path-sum
method:

* the Gauss hypergeometric function from SciPy, in the case where Hl
  reduces to it exactly;
* the power series of Hl about z = 0, summed here independently of the
  module, inside its disc of convergence;
* direct integration of Heun's equation with SciPy's ODE solver, outside
  that disc.

The high-precision checks of the power series need mpmath and are skipped
if it is not installed.
"""
# The series that starts the solution is a private helper, tested directly.
# pylint: disable=protected-access

import functools
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1  # pylint: disable=no-name-in-module

import heun_path_sum
from heun_path_sum import heun, subdivide_domain

PARAMETER_NAMES = ("a", "q", "alpha", "beta", "gamma", "delta")

# Parameters for which Hl is the hypergeometric function 2F1(0.7, 1.1; 1.6; z):
# epsilon = 0 and q = a*alpha*beta, for any a.
ALPHA, BETA, GAMMA = 0.7, 1.1, 1.6

# The second example of example.py: gamma = delta = epsilon = 1/2.
LAME = {"a": -19.0954, "q": 4.84052, "alpha": -1.0, "beta": 1.5,
        "gamma": 0.5, "delta": 0.5}

# A parameter set with a negative gamma.
NEGATIVE_GAMMA = {"a": 4.3, "q": -0.2, "alpha": 1.3, "beta": 0.12,
                  "gamma": -0.4, "delta": 4.32}


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------

def hypergeometric_case(a=2.5):
    """Heun parameters for which Hl(z) = 2F1(ALPHA, BETA; GAMMA; z)."""
    return {"a": a, "q": a*ALPHA*BETA, "alpha": ALPHA, "beta": BETA,
            "gamma": GAMMA, "delta": ALPHA + BETA + 1 - GAMMA}


def exact_hypergeometric(z):
    """The exact value of Hl in the hypergeometric case."""
    return hyp2f1(ALPHA, BETA, GAMMA, z)


def exact_hypergeometric_slope(z):
    """The exact derivative of Hl in the hypergeometric case."""
    return ALPHA*BETA/GAMMA*hyp2f1(ALPHA + 1, BETA + 1, GAMMA + 1, z)


def series_reference(z, *, a, q, alpha, beta, gamma, delta):
    # pylint: disable=too-many-arguments,too-many-locals
    """Hl at the points z from its power series, for abs(z) < min(1, abs(a)).

    Written independently of the module: the coefficients c_n are built
    first and the polynomial is then evaluated by Horner's rule.
    """
    z = np.asarray(z, dtype=complex)
    epsilon = alpha + beta + 1 - gamma - delta
    ratio = np.max(np.abs(z))/min(1, abs(a))
    terms = int(np.log(1e-18)/np.log(ratio)) + 20

    coefficients = [1.0, q/(a*gamma)]
    for n in range(1, terms):
        p_n = (n - 1 + alpha)*(n - 1 + beta)
        q_n = n*((n - 1 + gamma)*(1 + a) + a*delta + epsilon)
        r_n = a*(n + 1)*(n + gamma)
        coefficients.append(
            ((q_n + q)*coefficients[-1] - p_n*coefficients[-2])/r_n)

    total = np.zeros_like(z)
    for coefficient in reversed(coefficients):
        total = total*z + coefficient
    return total


def integrate_heun_equation(z_start, z_end, value, slope, *, a, q, alpha,
                            beta, gamma, delta):
    # pylint: disable=too-many-arguments
    """Integrate Heun's equation along the line from z_start to z_end.

    Returns the solution at z_end, given its value and slope at z_start.
    """
    epsilon = alpha + beta + 1 - gamma - delta
    direction = z_end - z_start

    def right_hand_side(t, state):
        z = z_start + t*direction
        first = gamma/z + delta/(z - 1) + epsilon/(z - a)
        zeroth = (alpha*beta*z - q)/(z*(z - 1)*(z - a))
        return [state[1]*direction,
                -(first*state[1] + zeroth*state[0])*direction]

    solution = solve_ivp(right_hand_side, (0.0, 1.0),
                         np.array([value, slope], dtype=complex),
                         method="DOP853", rtol=1e-12, atol=1e-14)
    assert solution.success
    return solution.y[0, -1]


def largest_relative_error(computed, exact):
    """Largest pointwise relative error.  For functions without zeros."""
    exact = np.asarray(exact)
    return float(np.max(np.abs(np.asarray(computed) - exact)/np.abs(exact)))


def error_relative_to_size(computed, exact):
    """Largest absolute error divided by the largest value of abs(exact).

    Used where Hl may pass through zero on the grid, which makes the
    pointwise relative error large for any method.
    """
    exact = np.asarray(exact)
    return float(np.max(np.abs(np.asarray(computed) - exact))
                 / np.max(np.abs(exact)))


def report(description, measured, documented=None):
    """Print a measured figure (shown by pytest -s), with the README's."""
    line = f"{description}: {measured:.1e}"
    if documented is not None:
        line += f"   (README: {documented:.1e})"
    # Start on a new line so that pytest's progress marks do not run into it.
    print("\n" + line, end=" ")


@functools.lru_cache(maxsize=None)
def hypergeometric_error(first_point, points):
    """Largest relative error of heun() on linspace(first_point, 0.95)."""
    z = np.linspace(first_point, 0.95, points)
    return largest_relative_error(heun(z, **hypergeometric_case()),
                                  exact_hypergeometric(z))


def solution_about_one(z):
    """The solution of the LAME equation that is analytic at z = 1."""
    return heun(1 - z, a=1 - LAME["a"],
                q=LAME["alpha"]*LAME["beta"] - LAME["q"],
                alpha=LAME["alpha"], beta=LAME["beta"],
                gamma=LAME["delta"], delta=LAME["gamma"])


def end_error_about_one(z):
    """Relative error of solution_about_one at the last point of z.

    The reference integrates the ORIGINAL equation in z, starting from the
    value and slope at z[0], so it also checks the change of variable.
    """
    value, slope = heun_path_sum._start_values(
        1 - z[0], complex(1 - LAME["a"]),
        complex(LAME["alpha"]*LAME["beta"] - LAME["q"]),
        complex(LAME["alpha"]), complex(LAME["beta"]),
        complex(LAME["delta"]), complex(LAME["gamma"]))
    exact = integrate_heun_equation(z[0], z[-1], value, -slope, **LAME)
    return abs(solution_about_one(z)[-1] - exact)/abs(exact)


def random_cases(count=60, seed=2026):
    """Reproducible random parameter sets and directions.

    A third of the cases run along the positive real axis, a third along
    the negative real axis, and a third along a complex ray with complex
    parameters.  Every fourth case has a negative real part of gamma.
    """
    # RandomState is used because its stream is frozen across NumPy versions.
    rng = np.random.RandomState(seed)  # pylint: disable=no-member
    cases = []
    for index in range(count):
        is_complex = index % 3 == 2

        def draw(low, high, is_complex=is_complex):
            value = rng.uniform(low, high)
            if is_complex:
                return complex(value, rng.uniform(-0.5, 0.5))
            return value

        if is_complex:
            # A phase within 45 degrees keeps a - z off the negative real
            # axis for abs(z) < 1, so no grid crosses a branch cut.
            a = rng.uniform(1.5, 5.0)*np.exp(
                1j*rng.uniform(-np.pi/4, np.pi/4))
            direction = np.exp(1j*rng.uniform(-np.pi, np.pi))
        else:
            a = rng.uniform(1.5, 5.0)*rng.choice([-1.0, 1.0])
            direction = 1.0 if index % 3 == 0 else -1.0

        gamma = draw(-0.9, -0.1) if index % 4 == 3 else draw(0.3, 3.0)
        parameters = {"a": a, "q": draw(-2, 2), "alpha": draw(-2, 2),
                      "beta": draw(-2, 2), "gamma": gamma,
                      "delta": draw(-2, 3)}
        cases.append((parameters, direction))
    return cases


def random_case_errors(points):
    """Error of heun() against the power series for every random case."""
    errors = []
    for parameters, direction in random_cases():
        z = np.linspace(0.05, 0.8, points)*direction
        sample = np.unique(np.linspace(1, points - 1, 25).astype(int))
        errors.append(error_relative_to_size(
            heun(z, **parameters)[sample],
            series_reference(z[sample], **parameters)))
    return np.array(errors)


# --------------------------------------------------------------------------
# The references themselves
# --------------------------------------------------------------------------

def test_series_reference_matches_hypergeometric():
    """The independent power series agrees with SciPy's 2F1."""
    z = np.array([0.01, 0.3, 0.6, 0.9, -0.5, -0.9, 0.3 + 0.4j])
    error = largest_relative_error(
        series_reference(z, **hypergeometric_case()),
        exact_hypergeometric(z))
    assert error < 1e-12


def test_series_reference_matches_high_precision():
    """The independent power series agrees with a 30-digit evaluation."""
    mpmath = pytest.importorskip("mpmath")
    for parameters, direction in random_cases(count=12):
        z = 0.7*direction
        with mpmath.workdps(30):
            exact = complex(_high_precision_series(mpmath, z, parameters)[0])
        computed = series_reference([z], **parameters)[0]
        assert abs(computed - exact) < 1e-11*max(1, abs(exact))


def _high_precision_series(mpmath, z, parameters):
    # pylint: disable=too-many-locals
    """Value and slope of Hl at z, summed in mpmath's working precision."""
    a, q, alpha, beta, gamma, delta = (
        mpmath.mpmathify(parameters[name]) for name in PARAMETER_NAMES)
    z = mpmath.mpmathify(z)
    epsilon = alpha + beta + 1 - gamma - delta
    previous, current = mpmath.mpf(1), q/(a*gamma)
    value, slope, power = previous + current*z, current, z
    for n in range(1, 5000):
        p_n = (n - 1 + alpha)*(n - 1 + beta)
        q_n = n*((n - 1 + gamma)*(1 + a) + a*delta + epsilon)
        r_n = a*(n + 1)*(n + gamma)
        previous, current = current, ((q_n + q)*current - p_n*previous)/r_n
        slope += (n + 1)*current*power
        power *= z
        value += current*power
        if n > 30 and abs(current*power) < mpmath.mpf(10)**(-28):
            break
    return value, slope


# --------------------------------------------------------------------------
# Start values: the power series of the module
# --------------------------------------------------------------------------

@pytest.mark.parametrize("z0", [1e-11, 1e-3, 0.1, 0.5, 0.9, 0.99, -0.5,
                                -0.9, 0.3 + 0.4j, -0.2 - 0.7j])
def test_start_values_match_hypergeometric(z0):
    """Value and slope agree with the exact 2F1 and its derivative."""
    parameters = [complex(hypergeometric_case()[name])
                  for name in PARAMETER_NAMES]
    value, slope = heun_path_sum._start_values(z0, *parameters)
    assert value == pytest.approx(exact_hypergeometric(z0), rel=1e-12)
    assert slope == pytest.approx(exact_hypergeometric_slope(z0), rel=1e-12)


def test_start_values_with_small_a():
    """The recurrence does not overflow when abs(a) is small.

    Hl is the same 2F1 for any a in the hypergeometric case, but the series
    coefficients grow like abs(a)**(-n).
    """
    parameters = [complex(hypergeometric_case(a=0.01)[name])
                  for name in PARAMETER_NAMES]
    for z0 in (0.005, 0.0099):
        value, slope = heun_path_sum._start_values(z0, *parameters)
        assert value == pytest.approx(exact_hypergeometric(z0), rel=1e-11)
        assert slope == pytest.approx(exact_hypergeometric_slope(z0),
                                      rel=1e-11)


def test_start_values_match_high_precision():
    """Value and slope agree with a 30-digit sum for random parameters."""
    mpmath = pytest.importorskip("mpmath")
    for parameters, direction in random_cases():
        z0 = 0.6*direction
        with mpmath.workdps(30):
            exact_value, exact_slope = (
                complex(number) for number
                in _high_precision_series(mpmath, z0, parameters))
        value, slope = heun_path_sum._start_values(
            z0, *(complex(parameters[name]) for name in PARAMETER_NAMES))
        assert abs(value - exact_value) < 1e-10*max(1, abs(exact_value))
        assert abs(slope - exact_slope) < 1e-10*max(1, abs(exact_slope))


def test_start_values_of_a_constant_solution():
    """With q = 0 and alpha = 0 the solution is Hl = 1 exactly."""
    value, slope = heun_path_sum._start_values(
        0.4, 2.5 + 0j, 0j, 0j, 1.1 + 0j, 1.6 + 0j, 0.5 + 0j)
    assert value == 1
    assert slope == 0


def test_start_values_too_close_to_the_edge():
    """The series is not summed where it converges too slowly."""
    parameters = [complex(hypergeometric_case()[name])
                  for name in PARAMETER_NAMES]
    with pytest.raises(ValueError, match="did not converge"):
        heun_path_sum._start_values(0.9999, *parameters)


# --------------------------------------------------------------------------
# Accuracy: the figures quoted in the README
# --------------------------------------------------------------------------

README_TABLE = {
    # first point: errors for 2,000, 5,000, 20,000 and 80,000 points
    0.3: (3.1e-07, 7.3e-08, 1.4e-08, 3.4e-09),
    0.1: (2.5e-06, 4.4e-07, 4.4e-08, 6.9e-09),
    0.03: (1.6e-05, 2.3e-06, 1.6e-07, 1.5e-08),
    0.01: (1.0e-04, 9.9e-06, 5.1e-07, 3.7e-08),
    0.003: (9.0e-04, 7.1e-05, 2.1e-06, 1.2e-07),
    0.001: (6.5e-03, 5.3e-04, 1.1e-05, 3.8e-07),
}
README_TABLE_POINTS = (2000, 5000, 20000, 80000)


@pytest.mark.parametrize("first_point", list(README_TABLE))
@pytest.mark.parametrize("column", range(len(README_TABLE_POINTS)))
def test_readme_accuracy_table(first_point, column):
    """Each cell of the README's accuracy table is reproduced."""
    points = README_TABLE_POINTS[column]
    documented = README_TABLE[first_point][column]
    measured = hypergeometric_error(first_point, points)
    report(f"2F1 check, first point {first_point}, {points} points",
           measured, documented)
    assert measured == pytest.approx(documented, rel=0.1)


@pytest.mark.parametrize("first_point", list(README_TABLE))
def test_error_falls_with_refinement(first_point):
    """Four times the points gives at least four times less error."""
    coarse, fine, finest = (hypergeometric_error(first_point, points)
                            for points in (5000, 20000, 80000))
    assert coarse/fine > 4
    assert fine/finest > 4


def test_readme_example_output():
    """The error printed by the first example of example.py."""
    measured = hypergeometric_error(0.05, 20000)
    report("example.py, hypergeometric case", measured, 9.0e-08)
    assert measured == pytest.approx(9.0e-08, rel=0.1)


def test_readme_unresolved_start():
    """A step of twice the first point gives a poor result, and a warning."""
    z = np.linspace(1e-4, 0.95, 5000)
    with pytest.warns(RuntimeWarning, match="start is not resolved"):
        computed = heun(z, **hypergeometric_case())
    measured = largest_relative_error(computed, exact_hypergeometric(z))
    report("unresolved start (first point 1e-4, 5000 points)", measured,
           2.3e-02)
    assert measured == pytest.approx(2.3e-02, rel=0.1)


@pytest.mark.parametrize("first_point, documented", [(0.001, 1.3e-03),
                                                     (0.05, 1.2e-06)])
def test_readme_negative_gamma(first_point, documented):
    """A close start costs more accuracy when gamma is negative."""
    z = np.linspace(first_point, 0.9, 20000)
    sample = np.unique(np.linspace(1, z.size - 1, 40).astype(int))
    measured = largest_relative_error(
        heun(z, **NEGATIVE_GAMMA)[sample],
        series_reference(z[sample], **NEGATIVE_GAMMA))
    report(f"gamma = -0.4, first point {first_point}", measured, documented)
    assert measured == pytest.approx(documented, rel=0.15)


@pytest.mark.parametrize("last_point, points, documented",
                         [(3, 20000, 3.9e-07), (21, 199501, 1.5e-05)])
def test_readme_long_intervals(last_point, points, documented):
    """Error at the end of the second example, and of a longer interval.

    This also checks the change of variable about z = 1, because the
    reference integrates the original equation.
    """
    measured = end_error_about_one(np.linspace(1.05, last_point, points))
    report(f"solution about z = 1, up to z = {last_point}", measured,
           documented)
    assert measured == pytest.approx(documented, rel=0.15)


@pytest.mark.parametrize("points, documented_median, documented_largest",
                         [(20000, 8.1e-08, 2.8e-07), (4000, 1.6e-06, 5.6e-06)])
def test_readme_random_parameter_sets(points, documented_median,
                                      documented_largest):
    """Median and largest error over the random parameter sets.

    The error is taken relative to the largest value of abs(Hl) on the
    grid, because some of these solutions pass through zero.
    """
    errors = random_case_errors(points)
    report(f"{errors.size} random sets, {points} points, median",
           float(np.median(errors)), documented_median)
    report(f"{errors.size} random sets, {points} points, largest",
           float(errors.max()), documented_largest)
    assert np.median(errors) == pytest.approx(documented_median, rel=0.1)
    assert errors.max() == pytest.approx(documented_largest, rel=0.1)


def test_constant_solution():
    """With q = 0 and alpha = 0 the result stays close to Hl = 1."""
    z = np.linspace(0.05, 0.9, 8000)
    computed = heun(z, a=4.3, q=0, alpha=0, beta=0.12, gamma=1.6, delta=0.5)
    assert np.max(np.abs(computed - 1)) < 2e-6


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

def test_output_shape_and_type():
    """heun returns one complex number per grid point."""
    z = np.linspace(0.05, 0.9, 500)
    computed = heun(z, **hypergeometric_case())
    assert computed.shape == z.shape
    assert computed.dtype == np.complex128


def test_first_value_is_the_series_value():
    """The first output is the start value itself."""
    z = np.linspace(0.05, 0.9, 500)
    computed = heun(z, **hypergeometric_case())
    assert computed[0] == pytest.approx(exact_hypergeometric(0.05), rel=1e-13)


def test_real_and_complex_parameters_agree():
    """Plain floats give the same numbers as complex-typed parameters."""
    z = np.linspace(0.05, 0.9, 500)
    as_floats = heun(z, **hypergeometric_case())
    as_complex = heun(z, **{name: complex(value) for name, value
                            in hypergeometric_case().items()})
    assert np.array_equal(as_floats, as_complex)


def test_a_list_is_accepted():
    """The grid may be any sequence."""
    z = np.linspace(0.05, 0.9, 500)
    assert np.array_equal(heun(list(z), **hypergeometric_case()),
                          heun(z, **hypergeometric_case()))


def test_block_keywords_default_to_the_module_constants():
    """Passing the default block limits explicitly changes nothing."""
    z = np.linspace(0.05, 0.9, 500)
    explicit = heun(z, **hypergeometric_case(),
                    max_sub_points=heun_path_sum.DEFAULT_MAX_SUB_POINTS,
                    max_sub_width=heun_path_sum.DEFAULT_MAX_SUB_WIDTH)
    assert np.array_equal(explicit, heun(z, **hypergeometric_case()))


def test_negative_and_complex_directions():
    """Grids along the negative real axis and along a complex ray work."""
    for direction in (-1.0, np.exp(0.7j)):
        z = np.linspace(0.05, 0.9, 4000)*direction
        error = largest_relative_error(heun(z, **hypergeometric_case()),
                                       exact_hypergeometric(z))
        assert error < 1e-5


def test_no_warning_on_a_resolved_grid():
    """A sensible grid runs without any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        heun(np.linspace(0.05, 0.9, 500), **hypergeometric_case())


BAD_INPUT = {
    "fewer than three points":
        (np.linspace(0.05, 0.1, 2), {}, "at least 3 points"),
    "two-dimensional grid":
        (np.full((3, 3), 0.1), {}, "one-dimensional"),
    "non-uniform grid":
        (np.array([0.05, 0.06, 0.08, 0.09]), {}, "uniformly spaced"),
    "repeated point":
        (np.full(5, 0.1), {}, "non-zero step"),
    "not finite":
        (np.array([0.1, np.nan, 0.3]), {}, "finite"),
    "grid containing 1":
        (np.linspace(0.5, 1.5, 11), {}, "singular points"),
    "grid containing a":
        (np.array([0.25, 0.3, 0.35]), {"a": 0.3}, "singular points"),
    "first point outside the disc":
        (np.linspace(1.2, 3, 500), {}, "must lie inside"),
    "first point outside the disc of a small a":
        (np.linspace(0.4, 0.9, 500), {"a": 0.3}, "must lie inside"),
    "first point too close to the edge":
        (np.linspace(0.9999, 0.99995, 50), {}, "did not converge"),
    "complex grid crossing a branch cut":
        (np.linspace(0.05, 0.9, 500)*np.exp(0.7j), {"a": -2 + 0.3j},
         "negative real axis"),
    "a = 0":
        (np.linspace(0.05, 0.9, 500), {"a": 0}, "a must not be 0"),
    "gamma = 0":
        (np.linspace(0.05, 0.9, 500), {"gamma": 0}, "gamma must not be"),
    "gamma a negative integer":
        (np.linspace(0.05, 0.9, 500), {"gamma": -2.0}, "gamma must not be"),
    "blocks of fewer than three points":
        (np.linspace(0.05, 0.9, 500), {"max_sub_points": 2},
         "max_sub_points"),
    "step wider than half a block":
        (np.linspace(0.5, 4000.5, 11), {}, "max_sub_width"),
}


@pytest.mark.parametrize("case", list(BAD_INPUT))
def test_bad_input_is_rejected(case):
    """Unusable input raises a ValueError that says what is wrong."""
    grid, overrides, message = BAD_INPUT[case]
    with pytest.raises(ValueError, match=message):
        heun(grid, **{**hypergeometric_case(), **overrides})


def test_subdivide_domain_shares_end_points():
    """Blocks are limited in size and overlap by exactly one point."""
    domain = np.linspace(0, 1, 250)
    blocks = subdivide_domain(domain, max_sub_points=100)
    assert [len(block) for block in blocks] == [100, 100, 52]
    for earlier, later in zip(blocks, blocks[1:]):
        assert earlier[-1] == later[0]
    assert blocks[0][0] == domain[0]
    assert blocks[-1][-1] == domain[-1]


def test_subdivide_domain_limits_the_width():
    """No block spans more than max_sub_width."""
    domain = np.linspace(0, 100, 1001)
    blocks = subdivide_domain(domain, max_sub_points=1000, max_sub_width=7.5)
    assert max(abs(block[-1] - block[0]) for block in blocks) <= 7.5
    assert sum(len(block) - 1 for block in blocks) == len(domain) - 1


def test_subdivide_domain_edge_cases():
    """An empty grid gives no blocks; a block needs at least two points."""
    assert not subdivide_domain(np.array([]))
    with pytest.raises(ValueError, match="max_sub_points"):
        subdivide_domain(np.linspace(0, 1, 10), max_sub_points=1)
