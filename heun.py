"""A module for computing Heun functions via the BGT method"""
import time
import numpy as np
from scipy.linalg import solve_triangular
from matplotlib import pyplot as plt

N = 1000
IDENTITY_MATRIX = np.identity(N)
BUFFER = 1e-9
V = np.zeros(N, dtype=complex)
V[0] = 1


def first_deriv_coeff(z_range: np.ndarray,
                      a: float, gamma: float, delta: float, epsilon: float) -> np.ndarray:
    """The coefficient of the first derivative in the Heun equation"""
    return gamma/z_range + delta/(z_range - 1) + epsilon/(z_range - a)


def second_deriv_coeff(z_range: np.ndarray,
                       a: float, q: float,  alpha: float, beta: float) -> np.ndarray:
    """The coefficient of the zeroth derivative in the Heun equation"""
    return (alpha * beta * z_range - q) / (z_range * (z_range - 1) * (z_range - a))


def weight_func(z_range, a, gamma, delta, epsilon):
    """The factor in the integrand of J(z) which precedes X(z)"""
    return (z_range**gamma) * ((z_range - 1)**delta) * ((a - z_range)**epsilon) * np.exp(z_range)


def get_kernel_1(factor_1, factor_2, delta_z) -> np.ndarray:
    """Returns an array which is the matrix form of K_1"""
    integrand = factor_1 * factor_2
    consecutive_integrals = 0.5*(integrand[0:N-1] + integrand[1:N])
    consecutive_integrals = np.insert(consecutive_integrals, 0, 0, axis=0)

    cumulative_sums = np.cumsum(consecutive_integrals)

    kx, ky = np.meshgrid(cumulative_sums, cumulative_sums, sparse=False)
    kernel = kx - ky

    prefactor, _ = np.meshgrid(factor_2, factor_2, sparse=False)

    kernel = kernel/prefactor

    kernel = 1 + delta_z*kernel
    return kernel


def get_kernel_2(X, Q) -> np.ndarray:
    """Returns an array which is the matrix form of K_2"""
    kernel = np.zeros((N, N), dtype=complex)

    # Can we vectorise this?
    for i in range(N):
        kernel[i] = np.real(np.exp(-z)*np.exp(z[i])*X + Q)

    return kernel


def get_kernel(subscript: int, X, Y, Q, delta_z) -> np.ndarray:
    """Returns either K_1 or K_2 depending on the value of K_number.
    Returns the zero matrix if there is no K matrix associated with K_number"""
    if subscript == 1:
        return get_kernel_1(X, Y, delta_z)

    if subscript == 2:
        return get_kernel_2(X, Q)

    return np.zeros((N, N))


def get_greens_func(G_number: int,
                    X: np.ndarray, Y: np.ndarray, Q: np.ndarray, delta_z: float) -> np.ndarray:
    """Returns the G_1 or G_2 matrix, depending on K_number, as defined in the BGT method"""
    res_kernel = get_kernel(G_number, X, Y, Q, delta_z)

    diagonal = np.diag(np.diag(res_kernel))

    lhs = IDENTITY_MATRIX - delta_z*res_kernel + 0.5*delta_z*diagonal

    green = (1/delta_z)*(solve_triangular(lhs, V, trans=1))

    green[0] = 0

    return green


def heun(a, q, alpha, beta, gamma, delta, z_range):
    """Returns teh R matrix, whose first column approximates the solution to the Heun equation"""
    epsilon = alpha + beta + 1 - gamma - delta
    delta_z = (z_range[-1] - z_range[0])/N

    P = first_deriv_coeff(z_range, a, gamma, delta, epsilon)
    Q = second_deriv_coeff(z_range, a, q, alpha, beta)
    X = - P - Q - 1
    Y = weight_func(z_range, a, gamma, delta, epsilon)

    green_1 = get_greens_func(1, X, Y, Q, delta_z)

    int_G_1 = 0.5*delta_z*(green_1[0:N - 1] + green_1[1:N])
    int_G_1 = np.insert(int_G_1, 0, 0, axis=0)
    int_G_1 = np.cumsum(int_G_1)
    heun_function = init_val*(1 + int_G_1)

    green_2 = get_greens_func(2, X, Y, Q, delta_z)

    exp_G_2 = np.exp(-z)*green_2
    sum_exp_G_2 = 0.5*delta_z*(exp_G_2[1:N]+exp_G_2[0:N-1])
    sum_exp_G_2 = np.insert(sum_exp_G_2, 0, 0, axis=0)
    sum_exp_G_2 = np.cumsum(sum_exp_G_2)
    res_G_2 = np.exp(z)*sum_exp_G_2

    int_G_2 = 0.5*delta_z*(green_2[1:N] + green_2[0:N-1])
    int_G_2 = np.insert(int_G_2, 0, 0, axis=0)
    int_G_2 = np.cumsum(int_G_2)

    R2 = np.exp(z-z_range[0]) - 1 + res_G_2 - int_G_2

    heun_function += (init_slope - init_val) * R2
    return heun_function


if __name__ == "__main__":
    # Heun parameters
    q_param = 1.92837 * 1e6 + 1j*0
    a_param = -1.10193 * 1e8 + 1j*0
    alpha_param = -1 + 1j*0
    beta_param = 3/2 + 1j*0
    gamma_param = 0.5 + 1j*0
    delta_param = 0.5 + 1j*0

    # Boundary conditions
    init_val = 0
    init_slope = 1

    # Domain
    z_min = 1 + BUFFER
    # z_max = 8.64359*1e6
    z_max = 2
    z = np.linspace(z_min, z_max, N)

    start = time.perf_counter()
    y = heun(1 - a_param, alpha_param*beta_param - q_param,
             alpha_param, beta_param, delta_param, gamma_param, 1 - z)
    end = time.perf_counter()

    print(end - start)

    plt.plot(z, y)
    plt.show()
