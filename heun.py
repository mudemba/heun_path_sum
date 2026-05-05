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


def first_order_coeff(z: np.ndarray, a: float, gamma: float, delta: float, epsilon: float) -> np.ndarray:
    """The coefficient of the first derivative in the Heun equation"""
    return gamma/z + delta/(z - 1) + epsilon/(z - a)


def second_order_coeff(z: np.ndarray, a: float, q: float, alpha: float, beta: float) -> np.ndarray:
    """The coefficient of the zeroth derivative in the Heun equation"""
    return (alpha * beta * z - q) / (z * (z - 1) * (z - a))


def weight_function(z_val, a, gamma, delta, epsilon):
    """The factor in the integrand of J(z) which precedes X(z)"""
    return (z_val**gamma) * ((z_val - 1)**delta) * ((a - z_val)**epsilon) * np.exp(z_val)


def J_integrand(z_val, a, q, alpha, beta, gamma, delta, epsilon):
    """The integrand of the J(z) integral"""
    return Y(z_val, a, gamma, delta, epsilon) * X(z_val, a, q, alpha, beta, gamma, delta, epsilon)


def delta_J(domain, a, q, alpha, beta, gamma, delta, epsilon):
    """Returns a vector whose elements are the trapezoidal integrals over small distances delta_z"""
    integrands = J_integrand(domain, a, q, alpha, beta, gamma, delta, epsilon)
    integrals = 0.5 * (integrands[0:N-1] + integrands[1:N])
    integrals = np.insert(integrals, 0, 0, axis=0)
    return integrals


def get_K_1_matrix(alpha, beta, gamma, delta, epsilon, z_interval) -> np.ndarray:
    """Returns an array which is the matrix form of K_1"""
    delta_J_vector = delta_J(z_interval, a, q, alpha,
                             beta, gamma, delta, epsilon)

    J_0_to_i = np.cumsum(delta_J_vector)

    K_1x, K_1y = np.meshgrid(J_0_to_i, J_0_to_i, sparse=False)
    K_1_matrix = K_1x - K_1y

    prefactors = Y(z, a, gamma, delta, epsilon)
    prefactor, _ = np.meshgrid(prefactors, prefactors, sparse=False)

    K_1_matrix = K_1_matrix/prefactor

    K_1_matrix = 1 + delta_z*K_1_matrix
    return K_1_matrix


def get_K_2_matrix() -> np.ndarray:
    """Returns an array which is the matrix form of K_2"""
    K_2_matrix = np.zeros((N, N), dtype=complex)

    # Can we vectorise this?
    for i in range(N):
        K_2_matrix[i] = np.real(
            np.exp(-z)*np.exp(z[i])*X(z[i], a, q, alpha, beta, gamma, delta, epsilon) + Q(z[i], a, q, alpha, beta))

    return K_2_matrix


def get_K_matrix(K_number: int) -> np.ndarray:
    """Returns either K_1 or K_2 depending on the value of K_number.
    Returns the zero matrix if there is no K matrix associated with K_number"""
    if K_number == 1:
        return get_K_1_matrix()

    if K_number == 2:
        return get_K_2_matrix()

    return np.zeros((N, N))


def get_G_column(G_number: int):
    """Returns the G_1 or G_2 matrix, depending on K_number, as defined in the BGT method"""
    K = get_K_matrix(G_number)

    dK = np.diag(np.diag(K))

    lhs = IDENTITY_MATRIX - delta_z*K + 0.5*delta_z*dK

    G = (1/delta_z)*(solve_triangular(lhs, V, trans=1))

    G[0] = 0

    return G


def Heun(a_param, q_param, alpha_param, beta_param, gamma_param, delta_param, z_interval):
    """Returns teh R matrix, whose first column approximates the solution to the Heun equation"""
    epsilon_param = alpha_param + beta_param + 1 - gamma_param - delta_param

    P = first_order_coeff(z_interval, a_param, gamma_param,
                          delta_param, epsilon_param)
    Q = second_order_coeff(z_interval, a_param, q_param,
                           alpha_param, beta_param)
    X = - P - Q - 1
    Y = weight_function(z_interval, a, gamma, delta, epsilon_param)

    G_1 = get_G_column(1)

    delta_z = (z_max - z_min)/N

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
