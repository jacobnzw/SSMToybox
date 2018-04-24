import numpy as np
import numpy.linalg as la
from bq.bqkern import RBF
from numpy import newaxis as na
from utils import maha
from mtran import Unscented


"""
There are two forms of the RBF kernel expectation E[k(x, x_i)k(x,x_j)] found in the literature. I suspect the 
simplified form given in M. Deisenroth's thesis might be wrong. The only contentious part of the expressions are the 
exponents, which are compared in this experiment.
"""


def exp_x_kxkx(par_0, par_1, x, scaling=False):
    """
    Correlation matrix of kernels with elements

    .. math:
    \[
        \mathbb{E}[k(x, x_i), k(x, x_j)] = \int\! k(x, x_i), k(x, x_j) N(x \mid 0, I)\, \mathrm{d}x
    \]

    Parameters
    ----------
    x : numpy.ndarray
        Data points, shape (D, N)
    par_0 : numpy.ndarray
    par_1 : numpy.ndarray
        Kernel parameters, shape (D, )
    scaling : bool
        Kernel scaling parameter used when `scaling=True`.

    Returns
    -------
    : numpy.ndarray
        Correlation matrix of kernels computed for given pair of kernel parameters.
    """

    # unpack kernel parameters
    alpha, sqrt_inv_lam = RBF._unpack_parameters(par_0)
    alpha_1, sqrt_inv_lam_1 = RBF._unpack_parameters(par_1)
    alpha, alpha_1 = (1.0, 1.0) if not scaling else (alpha, alpha_1)
    inv_lam = sqrt_inv_lam ** 2
    inv_lam_1 = sqrt_inv_lam_1 ** 2

    # \xi_i^T * \Lambda_m * \xi_i
    xi = sqrt_inv_lam.dot(x)  # (D, N)
    xi = np.sum(xi * xi, axis=0)  # (N, )

    # \xi_j^T * \Lambda_n * \xi_j
    xi_1 = sqrt_inv_lam_1.dot(x)  # (D, N)
    xi_1 = np.sum(xi_1 * xi_1, axis=0)  # (N, )

    # \Lambda^{-1} * x
    x_0 = inv_lam.dot(x)  # (D, N)
    x_1 = inv_lam_1.dot(x)

    # R^{-1} = (\Lambda_m^{-1} + \Lambda_n^{-1} + \eye)^{-1}
    r = inv_lam + inv_lam_1 + np.eye(x_0.shape[0])  # (D, D)

    n = (xi[:, na] + xi_1[na, :]) - maha(x_0.T, -x_1.T, V=la.inv(r))  # (N, N)
    # return la.det(r) ** -0.5 * np.exp(n)
    return n


def exp_x_kxkx_1(par_0, par_1, x):
    """
    Correlation matrix of kernels with elements

    .. math:
    \[
        \mathbb{E}[k(x, x_i), k(x, x_j)] = \int\! k(x, x_i), k(x, x_j) N(x \mid 0, I)\, \mathrm{d}x
    \]

    Parameters
    ----------
    x : numpy.ndarray
        Data points, shape (D, N)
    par_0 : numpy.ndarray
    par_1 : numpy.ndarray
        Kernel parameters, shape (D, )
    scaling : bool
        Kernel scaling parameter used when `scaling=True`.

    Returns
    -------
    : numpy.ndarray
        Correlation matrix of kernels computed for given pair of kernel parameters.
    """

    # unpack kernel parameters
    alpha, sqrt_inv_lam = RBF._unpack_parameters(par_0)
    alpha_1, sqrt_inv_lam_1 = RBF._unpack_parameters(par_1)
    inv_lam = sqrt_inv_lam ** 2
    inv_lam_1 = sqrt_inv_lam_1 ** 2

    # (x_i - x_j)\T (2\Lam)\I (x_i -x_j)
    # dx = x[..., na] - x[:, na, :]
    # dx = np.einsum('ij, jkl', (1/np.sqrt(2)) * sqrt_inv_lam, dx)
    # xi = sqrt_inv_lam.dot(dx)  # (D, N)
    # xi = 0.5 * np.sum(dx * dx, axis=0)  # (N, )

    # \xi_j^T * \Lambda_n * \xi_j
    # xi_1 = sqrt_inv_lam_1.dot(x)  # (D, N)
    # xi_1 = 2 * np.log(alpha_1) - 0.5 * np.sum(xi_1 * xi_1, axis=0)  # (N, )

    # \Lambda^{-1} * x
    # x = inv_lam.dot(x)  # (D, N)

    # R^{-1} = (\Lambda_m^{-1} + \Lambda_n^{-1} + \eye)^{-1}
    r = 0.5*la.inv(inv_lam) + np.eye(x.shape[0])  # (D, D)

    n = 0.5*maha(x.T, x.T, V=inv_lam) + 0.25*maha(x.T, -x.T, V=la.inv(r))  # (N, N)
    # return la.det(r) ** -0.5 * np.exp(n)
    return n


def exp_test(par_0, par_1, x):
    alpha, sqrt_inv_lam = RBF._unpack_parameters(par_0)
    alpha_1, sqrt_inv_lam_1 = RBF._unpack_parameters(par_1)
    inv_lam = sqrt_inv_lam ** 2
    inv_lam_1 = sqrt_inv_lam_1 ** 2

    A = 0.5 * maha(x.T, x.T, V=inv_lam)
    n = A.shape[0]
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = x[:, i] - x[:, j]
            B[i, j] = dx.T.dot(0.5*inv_lam).dot(dx)

    print('{}'.format(np.array_equal(A, B)))


# random points
dim, num_pts = 1, 3
# x = np.random.rand(dim, num_pts)
x = (1/np.sqrt(3))*Unscented.unit_sigma_points(dim)

# x = np.array([[0, 1]])

# kernel parameters
par = np.array([[1] + dim*[1]])

# Deiseroth's form
Q0 = exp_x_kxkx(par, par, x)

# Obvious form
Q1 = exp_x_kxkx_1(par, par, x)

exp_test(par, par, x)

print(Q0)
print('Symmetric: {}'.format(np.allclose(Q0, Q0.T)))
print(Q1)
print('Symmetric: {}'.format(np.allclose(Q1, Q1.T)))
print('Equal {}'.format(np.allclose(Q0, Q1)))