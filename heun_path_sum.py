"""A module for computing Heun functions via the BGT path sum method"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.integrate import cumulative_trapezoid


def heun_eq_coeff_1(z_range: np.ndarray,
                    a: complex, gamma: complex, delta: complex, epsilon: complex) -> np.ndarray:
    """The coefficient of the first-order derivative in the Heun equation"""
    return gamma/z_range + delta/(z_range - 1) + epsilon/(z_range - a)


def heun_eq_coeff_0(z_range: np.ndarray,
                    a: complex, q: complex,  alpha: complex, beta: complex) -> np.ndarray:
    """The coefficient of the zeroth-order derivative in the Heun equation"""
    return (alpha * beta * z_range - q) / (z_range * (z_range - 1) * (z_range - a))


def weight_func(z_range, a, gamma, delta, epsilon):
    """The factor in the integrand of J(z) which precedes X(z)"""
    return (z_range**gamma) * ((z_range - 1)**delta) * ((a - z_range)**epsilon) * np.exp(z_range)


def get_kernel_1(x_vec, y_vec, delta_z) -> np.ndarray:
    """Returns an array which is the matrix form of K_1"""
    integrand = x_vec * y_vec

    cumulative_sums = cumulative_trapezoid(integrand, dx=delta_z, initial=0)

    kx, ky = np.meshgrid(cumulative_sums, cumulative_sums, sparse=False)

    kernel = kx - ky

    divisor, _ = np.meshgrid(y_vec, y_vec, sparse=False)

    kernel = 1 + kernel/divisor

    return kernel


def get_kernel_2(x_vec, q_vec, z_range, points) -> np.ndarray:
    """Returns an array which is the matrix form of K_2"""
    exp_vec = np.exp(z_range)

    x_times_e, inv_exp = np.meshgrid(x_vec * exp_vec, 1/exp_vec)
    q_rows, _ = np.meshgrid(q_vec, q_vec, sparse=False)

    kernel = x_times_e*inv_exp + q_rows

    return kernel


def get_neumann_sum(matrix_kernel, delta_z, points):
    """Returns"""
    diagonal = np.diag(np.diag(matrix_kernel))
    identity = np.identity(points)
    v = np.zeros(points)
    v[0] = 1

    lhs = identity - delta_z*matrix_kernel + 0.5*delta_z*diagonal

    neumann_sum = (1/delta_z)*(solve_triangular(lhs, v, trans=1))

    neumann_sum[0] = 0

    return neumann_sum


def path_ordered_exp_1(x_vec: np.ndarray, y_vec: np.ndarray, delta_z: float, points) -> np.ndarray:
    """Returns the first contribution to the path-ordered exponential.
    Computation chain: kernel K_1 -> Green's function G_1 -> path-ordered exponential U_11"""
    kernel = get_kernel_1(x_vec, y_vec, delta_z)
    green = get_neumann_sum(kernel, delta_z, points)

    int_g_1 = 0.5*delta_z*(green[0:points - 1] + green[1:points])
    int_g_1 = np.insert(int_g_1, 0, 0, axis=0)
    int_g_1 = np.cumsum(int_g_1)
    return 1 + int_g_1


def path_ordered_exp_2(q_vec: np.ndarray, x_vec: np.ndarray,
                       delta_z: float, z_range: np.ndarray, points) -> np.ndarray:
    """Returns the second contribution to the path-ordered exponential.
    Computation chain: kernel K_2 -> Green's function G_2 -> path-ordered exponential U_12"""
    kernel_2 = get_kernel_2(x_vec, q_vec, z_range, points)
    green_2 = get_neumann_sum(kernel_2, delta_z, points)

    exp_g_2 = np.exp(-z_range)*green_2
    sum_exp_g_2 = 0.5*delta_z*(exp_g_2[1:points]+exp_g_2[0:points-1])
    sum_exp_g_2 = np.insert(sum_exp_g_2, 0, 0, axis=0)
    sum_exp_g_2 = np.cumsum(sum_exp_g_2)
    res_g_2 = np.exp(z_range)*sum_exp_g_2

    int_g_2 = 0.5*delta_z*(green_2[1:points] + green_2[0:points-1])
    int_g_2 = np.insert(int_g_2, 0, 0, axis=0)
    int_g_2 = np.cumsum(int_g_2)

    res_2 = np.exp(z_range-z_range[0]) - 1 + res_g_2 - int_g_2

    return res_2


def heun(z_range: np.ndarray, *, a: complex, q: complex,
         alpha: complex, beta: complex, gamma: complex, delta: complex) -> np.ndarray:
    """Returns the R matrix, whose first column approximates the solution to the Heun equation"""
    epsilon = 1 + alpha + beta - gamma - delta

    points = len(z_range)
    delta_z = (z_range[-1] - z_range[0])/(points - 1)
    direction = np.sign(z_range[-1] - z_range[0])
    init_val, init_slope = 1, direction*q/(gamma*a)

    p_func = heun_eq_coeff_1(z_range, a, gamma, delta, epsilon)
    q_func = heun_eq_coeff_0(z_range, a, q, alpha, beta)
    x_func = - p_func - q_func - 1
    y_func = weight_func(z_range, a, gamma, delta, epsilon)

    heun_function = init_val*path_ordered_exp_1(
        x_func, y_func, delta_z, points)

    heun_function += (init_val - init_slope)*path_ordered_exp_2(q_func,
                                                                x_func, delta_z, z_range, points)

    return heun_function
