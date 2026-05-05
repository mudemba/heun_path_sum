"""A module for computing Heun functions via the BGT method"""

import numpy as np
from scipy.special import hyp2f1
from scipy.linalg import solve_triangular
from matplotlib import pyplot as plt
import time

N = 1000
IDENTITY_MATRIX = np.identity(N)
ZEROS_MATRIX = np.zeros((N, N), dtype=complex)
BUFFER = 1e-9
V = np.zeros(N, dtype=complex)
V[0] = 1


def P(z_val):
    """The coefficient of the first derivative in the Heun equation"""
    return gamma/z_val + delta/(z_val - 1) + epsilon/(z_val - a)


def Q(z_val):
    """The coefficient of the zeroth derivative in the Heun equation"""
    return (alpha * beta * z_val - q) / (z_val * (z_val - 1) * (z_val - a))


def X(z_val):
    """A factor appearing in the definition of both K_1 and K_2"""
    return - Q(z_val) - epsilon/(a - z_val) - gamma/z_val - delta/(z_val - 1) - 1


def Y(z_val):
    """The factor in the integrand of J(z) which precedes X(z)"""
    return (z_val**gamma) * ((z_val - 1)**delta) * ((a - z_val)**epsilon) * np.exp(z_val)


def J_integrand(z_val):
    """The integrand of the J(z) integral"""
    return Y(z_val) * X(z_val)


def delta_J(domain):
    """Returns a vector whose elements are the trapezoidal integrals over small distances delta_z"""
    integrands = J_integrand(domain)
    integrals = 0.5 * (integrands[0:N-1] + integrands[1:N])
    integrals = np.insert(integrals, 0, 0, axis=0)
    return integrals


def get_K_1_matrix() -> np.ndarray:
    """Returns an array which is the matrix form of K_1"""
    delta_J_vector = delta_J(z)

    J_0_to_i = np.cumsum(delta_J_vector)

    K_1x, K_1y = np.meshgrid(J_0_to_i, J_0_to_i, sparse=False)
    K_1_matrix = (K_1x - K_1y)

    prefactors = Y(z)
    prefactor, _ = np.meshgrid(prefactors, prefactors, sparse=False)

    K_1_matrix = K_1_matrix/prefactor

    K_1_matrix = 1 + delta_z*K_1_matrix
    return K_1_matrix


def get_K_2_matrix() -> np.ndarray:
    """Returns an array which is the matrix form of K_2"""
    K_2_matrix = np.zeros((N, N), dtype=complex)

    # Can we vectorise this?
    for i in range(N):
        K_2_matrix[i] = np.real(np.exp(-z)*np.exp(z[i])*X(z[i]) + Q(z[i]))

    return K_2_matrix


def get_K_matrix(K_number: int) -> np.ndarray:
    """Returns either K_1 or K_2 depending on the value of K_number.
    Returns the zero matrix if there is no K matrix associated with K_number"""
    if K_number == 1:
        return get_K_1_matrix()

    if K_number == 2:
        return get_K_2_matrix()

    return ZEROS_MATRIX


def get_G_column(G_number: int):
    """Returns the G_1 or G_2 matrix, depending on K_number, as defined in the BGT method"""
    K = get_K_matrix(G_number)

    dK = np.diag(np.diag(K))

    lhs = IDENTITY_MATRIX - delta_z*K + 0.5*delta_z*dK

    G = (1/delta_z)*(solve_triangular(lhs, V, trans=1))

    G[0] = 0

    return G


def Heun(a, q, alpha, beta, gamma, delta, z):
    """Returns teh R matrix, whose first column approximates the solution to the Heun equation"""
    G_1 = get_G_column(1)

    int_G_1 = 0.5*delta_z*(G_1[0:N - 1] + G_1[1:N])
    int_G_1 = np.insert(int_G_1, 0, 0, axis=0)
    int_G_1 = np.cumsum(int_G_1)
    Hl = Hl_0*(1 + int_G_1)

    G_2 = get_G_column(2)

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


def asymptotic_Heun(a, q, alpha, beta, gamma, delta, z):
    pass


if __name__ == "__main__":
    # Heun parameters
    q = 1.92837 * 1e6 + 1j*0
    a = -1.10193 * 1e8 + 1j*0
    alpha = -1 + 1j*0
    beta = 3/2 + 1j*0
    gamma = 0.5 + 1j*0
    delta = 0.5 + 1j*0
    epsilon = alpha + beta + 1 - gamma - delta

    # Boundary conditions
    Hl_0 = 0
    Hl_0_prime = 1

    # Domain
    z_min = 1 + BUFFER
    # z_max = 8.64359*1e6
    z_max = 2
    delta_z = (z_max - z_min)/N
    z = np.linspace(z_min, z_max, N)

    start = time.perf_counter()
    y = Heun(1 - a, alpha*beta - q, alpha, beta, delta, gamma, 1 - z)
    end = time.perf_counter()

    print(end - start)

    plt.plot(z, y)
    plt.show()
