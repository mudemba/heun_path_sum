"""A module for computing Heun functions via the BGT path sum method"""
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.integrate import cumulative_trapezoid
from matplotlib import pyplot as plt


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
    return (z_range**gamma) * ((z_range - 1)**delta) * ((a - z_range)**epsilon) * np.exp(z_range)


def get_kernel_1(x_vec: np.ndarray, y_vec: np.ndarray, delta_z: float) -> np.ndarray:
    """Returns an array which is the matrix form of K_1"""
    integrand = x_vec * y_vec

    cumulative_traps = cumulative_trapezoid(integrand, dx=delta_z, initial=0)

    kx, ky = np.meshgrid(cumulative_traps, cumulative_traps, sparse=False)

    kernel = kx - ky

    divisor, _ = np.meshgrid(y_vec, y_vec, sparse=False)

    kernel = 1 + kernel/divisor

    return kernel


def get_kernel_2(x_vec: np.ndarray, q_vec: np.ndarray, z_range: np.ndarray) -> np.ndarray:
    """Returns an array which is the matrix form of K_2"""
    exp_vec = np.exp(z_range)

    x_times_e, exp = np.meshgrid(x_vec * exp_vec, exp_vec, sparse=False)
    q_rows, _ = np.meshgrid(q_vec, q_vec, sparse=False)

    kernel = x_times_e/exp + q_rows

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


def path_ordered_exp_1(x_vec: np.ndarray, y_vec: np.ndarray,
                       delta_z: float, points: int) -> np.ndarray:
    """Returns the first contribution to the path-ordered exponential.
    Computation chain: kernel K_1 -> Green's function G_1 -> path-ordered exponential U_11"""
    kernel = get_kernel_1(x_vec, y_vec, delta_z)
    green = neumann_sum(kernel, delta_z, points)

    integral_of_green = cumulative_trapezoid(
        green, dx=delta_z, axis=0, initial=0)
    return 1 + integral_of_green


def path_ordered_exp_2(q_vec: np.ndarray, x_vec: np.ndarray,
                       delta_z: float, z_range: np.ndarray, points: int) -> np.ndarray:
    """Returns the second contribution to the path-ordered exponential.
    Computation chain: kernel K_2 -> Green's function G_2 -> path-ordered exponential U_12"""
    kernel = get_kernel_2(x_vec, q_vec, z_range)
    green = neumann_sum(kernel, delta_z, points)

    exp_green = np.exp(-z_range)*green
    int_part_1 = cumulative_trapezoid(
        exp_green, dx=delta_z, axis=0, initial=0)
    int_part_1 = np.exp(z_range)*int_part_1

    int_part_2 = cumulative_trapezoid(green, dx=delta_z, axis=0, initial=0)

    contribution = np.exp(z_range-z_range[0]) - 1 + int_part_1 - int_part_2

    return contribution


def heun(z_range: np.ndarray, *, a: complex, q: complex,
         alpha: complex, beta: complex, gamma: complex, delta: complex) -> np.ndarray:
    """Returns the R matrix, whose first column approximates the solution to the Heun equation"""
    epsilon = 1 + alpha + beta - gamma - delta

    z0 = z_range[0]
    points = len(z_range)
    delta_z = (z_range[-1] - z0)/(points - 1)
    init_val = 1 + q * z0/(gamma * a) + (z0**2) * (
        q**2 - a * alpha * beta + q *
        (1 + alpha + beta + a * gamma + delta * (a-1)))/(2 * a**2 * gamma * (1 + gamma))
    init_slope = q/(gamma*a) + z0*(q**2 - a*alpha*beta +
                                   q*(1+alpha+beta+a*gamma+delta*(a-1)))/(a**2*gamma*(1+gamma))

    p_func = heun_eq_coeff_1(z_range, a, gamma, delta, epsilon)
    q_func = heun_eq_coeff_0(z_range, a, q, alpha, beta)
    x_func = - p_func - q_func - 1
    y_func = weight_func(z_range, a, gamma, delta, epsilon)

    heun_function = init_val*path_ordered_exp_1(
        x_func, y_func, delta_z, points)

    heun_function += (init_slope - init_val)*path_ordered_exp_2(q_func,
                                                                x_func, delta_z, z_range, points)

    return heun_function


if __name__ == "__main__":
    # Heun parameters
    A = 4.3 + 1j*0
    Q = -0.2 + 1j*0
    ALPHA = 1.3 + 1j*0
    BETA = 0.12 + 1j*0
    GAMMA = -0.14 + 1j*0
    DELTA = 4.32 + 1j*0

    # Domain definition
    N = 100
    BUFFER = 1e-11

    Z_MIN = 0.1
    Z_MAX = 0.5
    Z = np.linspace(Z_MIN, Z_MAX, N)

    start = time.perf_counter()
    y = heun(Z, a=A, q=Q, alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA)
    end = time.perf_counter()

    print(end - start)

    plt.subplot(1, 2, 1)
    plt.plot(Z, np.real(y))
    plt.xlabel("$z$")
    plt.ylabel("$Re(Hl(z))$")

    plt.subplot(1, 2, 2)
    plt.plot(Z, np.imag(y))
    plt.xlabel("$z$")
    plt.ylabel("$Im(Hl(z))$")

    plt.show()
