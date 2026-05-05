"""A module for computing Heun functions via the BGT method"""
import numpy as np
from scipy.linalg import solve_triangular


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


def get_kernel_1(x_vec, y_vec, delta_z, points) -> np.ndarray:
    """Returns an array which is the matrix form of K_1"""
    integrand = x_vec * y_vec
    consecutive_integrals = 0.5*(integrand[0:points-1] + integrand[1:points])
    consecutive_integrals = np.insert(consecutive_integrals, 0, 0, axis=0)

    cumulative_sums = np.cumsum(consecutive_integrals)

    kx, ky = np.meshgrid(cumulative_sums, cumulative_sums, sparse=False)
    kernel = kx - ky

    prefactor, _ = np.meshgrid(y_vec, y_vec, sparse=False)

    kernel = kernel/prefactor

    kernel = 1 + delta_z*kernel
    return kernel


def get_kernel_2(x_vec, q_vec, z_range, points) -> np.ndarray:
    """Returns an array which is the matrix form of K_2"""
    kernel = np.zeros((points, points), dtype=complex)

    # Can we vectorise this?
    for i in range(points):
        kernel[i] = np.real(np.exp(-z_range)*np.exp(z_range[i])*x_vec + q_vec)

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


def path_ordered_exp_1(x_vec: np.ndarray, y_vec: np.ndarray, delta_z: float,
                       init_val, points) -> np.ndarray:
    """Returns the contribution to the path-ordered exponential from K_1"""
    kernel_1 = get_kernel_1(x_vec, y_vec, delta_z, points)
    green_1 = get_neumann_sum(kernel_1, delta_z, points)

    int_g_1 = 0.5*delta_z*(green_1[0:points - 1] + green_1[1:points])
    int_g_1 = np.insert(int_g_1, 0, 0, axis=0)
    int_g_1 = np.cumsum(int_g_1)
    return init_val*(1 + int_g_1)


def path_ordered_exp_2(q_vec: np.ndarray, x_vec: np.ndarray,
                       delta_z: float, z_range: np.ndarray,
                       init_val, init_slope, points) -> np.ndarray:
    """Returns the contribution to the path-ordered exponential from K_2"""
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

    return (init_slope - init_val) * res_2


def path_sum(init_conditions: list, params: list, z_range: np.ndarray) -> np.ndarray:
    """Returns the R matrix, whose first column approximates the solution to the Heun equation"""
    a, q = params[0], params[1]
    alpha, beta, gamma, delta = params[2], params[3], params[4], params[5]
    epsilon = 1 + alpha + beta - gamma - delta

    points = len(z_range)
    init_val, init_slope = init_conditions[0], init_conditions[1]
    delta_z = (z_range[-1] - z_range[0])/(points - 1)

    p_func = heun_eq_coeff_1(z_range, a, gamma, delta, epsilon)
    q_func = heun_eq_coeff_0(z_range, a, q, alpha, beta)
    x_func = - p_func - q_func - 1
    y_func = weight_func(z_range, a, gamma, delta, epsilon)

    heun_function = path_ordered_exp_1(
        x_func, y_func, delta_z, init_val, points)

    heun_function += path_ordered_exp_2(q_func,
                                        x_func, delta_z, z_range, init_val, init_slope, points)

    return heun_function
