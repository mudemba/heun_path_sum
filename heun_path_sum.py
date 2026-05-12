"""A module for computing Heun functions via the BGT path sum method"""
import time  # remove in final version
import numpy as np
from scipy.linalg import solve_triangular
from scipy.integrate import cumulative_trapezoid
from matplotlib import pyplot as plt  # remove in final version


def heun_eq_coeff_1(z_range: np.ndarray,
                    a: complex, gamma: complex, delta: complex, epsilon: complex) -> np.ndarray:
    """The coefficient of the first-order derivative in the Heun equation"""
    return gamma/z_range + delta/(z_range - 1) + epsilon/(z_range - a)


def heun_eq_coeff_0(z_range: np.ndarray,
                    a: complex, q: complex,  alpha: complex, beta: complex) -> np.ndarray:
    """The coefficient of the zeroth-order derivative in the Heun equation"""
    return (alpha * beta * z_range - q) / (z_range * (z_range - 1) * (z_range - a))


def weight_func(z_range: np.ndarray,
                a: complex, gamma: complex, delta: complex, epsilon: complex) -> np.ndarray:
    """The factor in the integrand of J(z) which precedes X(z)"""
    return (z_range**gamma) * ((z_range - 1)**delta) * ((a - z_range)**epsilon)


def get_kernel_1(z_range, x_vec: np.ndarray, y_vec: np.ndarray, delta_z: float, points: int) -> np.ndarray:
    """Returns an array which is the matrix form of K_1"""
    ky = np.zeros((points, points), dtype=complex)
    kernel = np.zeros((points, points), dtype=complex)
    integrand = x_vec * y_vec

    for j in range(points):
        ky = cumulative_trapezoid(
            integrand*np.exp(z_range - z_range[j]), dx=delta_z, initial=0)
        kernel[:, j] = ky[j] - ky

    divisor, _ = np.meshgrid(y_vec, y_vec, sparse=False)

    kernel = 1 + kernel/divisor

    return kernel


def get_kernel_2(x_vec: np.ndarray, q_vec: np.ndarray, z_range: np.ndarray, points: int) -> np.ndarray:
    """Returns an array which is the matrix form of K_2"""
    kernel = np.zeros((points, points), dtype=complex)

    for i in range(points):
        kernel[i] = x_vec*np.exp(z_range - z_range[i]) + q_vec

    return kernel


def neumann_sum(matrix_kernel: np.ndarray, delta_z: float, points: int) -> np.ndarray:
    """Returns the Neumann series of the kernel matrix_kernel,
    which is equivalent to the inverse *-resolvent of 1 - kernel"""
    diagonal = np.diag(np.diag(matrix_kernel))
    identity = np.identity(points)
    v = np.zeros(points)
    v[0] = 1

    lhs = identity - delta_z*matrix_kernel + 0.5*delta_z*diagonal

    n_sum = (1/delta_z)*(solve_triangular(lhs, v, trans=1) - v)

    return n_sum


def path_ordered_exp_1(z_range, x_vec: np.ndarray, y_vec: np.ndarray,
                       delta_z: float, points: int) -> np.ndarray:
    """Returns the first contribution to the path-ordered exponential.
    Computation chain: kernel K_1 -> Green's function G_1 -> path-ordered exponential U_11"""
    kernel = get_kernel_1(z_range, x_vec, y_vec, delta_z, points)
    green = neumann_sum(kernel, delta_z, points)

    integral_of_green = cumulative_trapezoid(
        green, dx=delta_z, axis=0, initial=0)
    return 1 + integral_of_green


def path_ordered_exp_2(q_vec: np.ndarray, x_vec: np.ndarray,
                       delta_z: float, z_range: np.ndarray, points: int) -> np.ndarray:
    """Returns the second contribution to the path-ordered exponential.
    Computation chain: kernel K_2 -> Green's function G_2 -> path-ordered exponential U_12"""
    kernel = get_kernel_2(x_vec, q_vec, z_range, points)
    green = neumann_sum(kernel, delta_z, points)

    int_part_1 = np.zeros((points, points), dtype=complex)
    for i in range(points):
        int_part_1[i] = cumulative_trapezoid(
            green*np.exp(z_range[i] - z_range), dx=delta_z, axis=0, initial=0)

    int_part_2 = cumulative_trapezoid(green, dx=delta_z, axis=0, initial=0)

    contribution = np.exp(
        z_range-z_range[0]) - 1 + np.diag(int_part_1) - int_part_2

    return contribution


def subdivide_domain(domain: np.ndarray, max_sub_points=100, max_sub_width=200) -> list[np.ndarray]:
    """ Splits an interval into subintervals, ensuring each subinterval contains no more than max_sub_points points and has a width no greater than max_sub_width."""
    if len(domain) == 0:
        return []

    subdomains = []
    start_idx = 0

    for i in range(1, len(domain)):
        would_exceed_points = (i - start_idx + 1) > max_sub_points
        would_exceed_width = (domain[i] - domain[start_idx]) > max_sub_width

        if would_exceed_points or would_exceed_width:
            # Save up to (not including) i
            subdomains.append(domain[start_idx:i])
            start_idx = i

    subdomains.append(domain[start_idx:])  # Append the final subinterval
    return subdomains


def heun(z_range: np.ndarray, *, a: complex, q: complex,
         alpha: complex, beta: complex, gamma: complex, delta: complex) -> np.ndarray:
    """Returns the R matrix, whose first column approximates the solution to the Heun equation"""
    epsilon = 1 + alpha + beta - gamma - delta

    z0 = z_range[0]
    init_val = 1 + q * z0/(gamma * a) + (z0**2) * (
        q**2 - a * alpha * beta + q *
        (1 + alpha + beta + a * gamma + delta * (a-1)))/(2 * a**2 * gamma * (1 + gamma))
    init_slope = q/(gamma*a) + z0*(q**2 - a*alpha*beta +
                                   q*(1+alpha+beta+a*gamma+delta*(a-1)))/(a**2*gamma*(1+gamma))

    total_points = len(z_range)
    delta_z = (z_range[-1] - z0)/(total_points - 1)

    subintervals = subdivide_domain(z_range)
    heun_function = np.array([])
    for subinterval in subintervals:
        points = len(subinterval)
        p_func = heun_eq_coeff_1(subinterval, a, gamma, delta, epsilon)
        q_func = heun_eq_coeff_0(subinterval, a, q, alpha, beta)
        x_func = - p_func - q_func - 1
        y_func = weight_func(subinterval, a, gamma, delta, epsilon)
        contribution = init_val*path_ordered_exp_1(subinterval,
                                                   x_func, y_func, delta_z, points)

        contribution += (init_slope - init_val)*path_ordered_exp_2(q_func,
                                                                   x_func, delta_z, subinterval, points)
        heun_function = np.append(heun_function, contribution)
        init_val = heun_function[-1]
        init_slope = (heun_function[-1] - heun_function[-2])/delta_z

    return heun_function


# remove in final version
if __name__ == "__main__":
    # Heun parameters
    A = 4.3 + 1j*0
    Q = -0.2 + 1j*0
    ALPHA = 1.3 + 1j*0
    BETA = 0.12 + 1j*0
    GAMMA = -0.14 + 1j*0
    DELTA = 4.32 + 1j*0

    # Domain definition
    N = 100000
    # BUFFER = 1e-11

    Z_MIN = 1.1
    Z_MAX = 1e5
    Z = np.linspace(Z_MIN, Z_MAX, N)

    start = time.perf_counter()
    y = heun(1-Z, a=A, q=Q, alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA)
    end = time.perf_counter()

    print(f"Evaluation completed in {end - start} seconds.")

    # plt.subplot(1, 2, 1)
    # plt.plot(Z, np.real(y))
    # plt.xlabel("$z$")
    # plt.ylabel("$Re(Hl(z))$")

    # plt.subplot(1, 2, 2)
    # plt.plot(Z, np.imag(y))
    # plt.xlabel("$z$")
    # plt.ylabel("$Im(Hl(z))$")

    # plt.show()
