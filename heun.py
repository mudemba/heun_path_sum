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


def first_derivative_coeff(z_range: np.ndarray, a: float, gamma: float, delta: float, epsilon: float) -> np.ndarray:
    """The coefficient of the first derivative in the Heun equation"""
    return gamma/z_range + delta/(z_range - 1) + epsilon/(z_range - a)


def second_derivative_coeff(z_range: np.ndarray, a: float, q: float, alpha: float, beta: float) -> np.ndarray:
    """The coefficient of the zeroth derivative in the Heun equation"""
    return (alpha * beta * z_range - q) / (z_range * (z_range - 1) * (z_range - a))


def weight_func(z_range, a, gamma, delta, epsilon):
    """The factor in the integrand of J(z) which precedes X(z)"""
    return (z_range**gamma) * ((z_range - 1)**delta) * ((a - z_range)**epsilon) * np.exp(z_range)


def kernel_1(X, Y, delta_z) -> np.ndarray:
    """Returns an array which is the matrix form of K_1"""
    integrand = X * Y
    consecutive_integrals = 0.5*(integrand[0:N-1] + integrand[1:N])
    consecutive_integrals = np.insert(consecutive_integrals, 0, 0, axis=0)

    cumulative_sums = np.cumsum(consecutive_integrals)

    kx, ky = np.meshgrid(cumulative_sums, cumulative_sums, sparse=False)
    kernel = kx - ky

    prefactor, _ = np.meshgrid(Y, Y, sparse=False)

    kernel = kernel/prefactor

    kernel = 1 + delta_z*kernel
    return kernel


def kernel_2(X, Q) -> np.ndarray:
    """Returns an array which is the matrix form of K_2"""
    kernel = np.zeros((N, N), dtype=complex)

    # Can we vectorise this?
    for i in range(N):
        kernel[i] = np.real(np.exp(-z)*np.exp(z[i])*X + Q)

    return kernel


def kernel(subscript: int, X, Y, Q, delta_z) -> np.ndarray:
    """Returns either K_1 or K_2 depending on the value of K_number.
    Returns the zero matrix if there is no K matrix associated with K_number"""
    if subscript == 1:
        return kernel_1(X, Y, delta_z)

    if subscript == 2:
        return kernel_2(X, Q)

    return np.zeros((N, N))


def greens_func(G_number: int, X: np.ndarray, Y: np.ndarray, Q: np.ndarray, delta_z: float) -> np.ndarray:
    """Returns the G_1 or G_2 matrix, depending on K_number, as defined in the BGT method"""
    res_kernel = kernel(G_number, X, Y, Q, delta_z)

    diagonal = np.diag(np.diag(res_kernel))

    lhs = IDENTITY_MATRIX - delta_z*res_kernel + 0.5*delta_z*diagonal

    green = (1/delta_z)*(solve_triangular(lhs, V, trans=1))

    green[0] = 0

    return green


def Heun(a_param, q_param, alpha_param, beta_param, gamma_param, delta_param, z_range):
    """Returns teh R matrix, whose first column approximates the solution to the Heun equation"""
    epsilon_param = alpha_param + beta_param + 1 - gamma_param - delta_param
    delta_z = (z_range[-1] - z_range[0])/N

    P = first_derivative_coeff(
        z_range, a_param, gamma_param, delta_param, epsilon_param)
    Q = second_derivative_coeff(
        z_range, a_param, q_param, alpha_param, beta_param)
    X = - P - Q - 1
    Y = weight_func(z_range, a_param, gamma_param, delta_param, epsilon_param)

    G_1 = greens_func(1, X, Y, Q, delta_z)

    delta_z = (z_max - z_min)/N

    int_G_1 = 0.5*delta_z*(G_1[0:N - 1] + G_1[1:N])
    int_G_1 = np.insert(int_G_1, 0, 0, axis=0)
    int_G_1 = np.cumsum(int_G_1)
    Hl = Hl_0*(1 + int_G_1)

    G_2 = greens_func(2, X, Y, Q, delta_z)

    exp_G_2 = np.exp(-z)*G_2
    sum_exp_G_2 = 0.5*delta_z*(exp_G_2[1:N]+exp_G_2[0:N-1])
    sum_exp_G_2 = np.insert(sum_exp_G_2, 0, 0, axis=0)
    sum_exp_G_2 = np.cumsum(sum_exp_G_2)
    res_G_2 = np.exp(z)*sum_exp_G_2

    int_G_2 = 0.5*delta_z*(G_2[1:N] + G_2[0:N-1])
    int_G_2 = np.insert(int_G_2, 0, 0, axis=0)
    int_G_2 = np.cumsum(int_G_2)

    R2 = np.exp(z-z_min) - 1 + res_G_2 - int_G_2

    Hl += (Hl_0_prime - Hl_0) * R2
    return Hl


if __name__ == "__main__":
    # Heun parameters
    q = 1.92837 * 1e6 + 1j*0
    a = -1.10193 * 1e8 + 1j*0
    alpha = -1 + 1j*0
    beta = 3/2 + 1j*0
    gamma = 0.5 + 1j*0
    delta = 0.5 + 1j*0

    # Boundary conditions
    Hl_0 = 0
    Hl_0_prime = 1

    # Domain
    z_min = 1 + BUFFER
    # z_max = 8.64359*1e6
    z_max = 2
    z = np.linspace(z_min, z_max, N)

    start = time.perf_counter()
    y = Heun(1 - a, alpha*beta - q, alpha, beta, delta, gamma, 1 - z)
    end = time.perf_counter()

    print(end - start)

    plt.plot(z, y)
    plt.show()
