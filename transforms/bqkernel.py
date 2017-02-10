from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve


# TODO: documentation

def maha(x, y, V=None):
    """
    Mahalanobis distance of all pairs of supplied data points.

    Parameters
    ----------
    x : numpy.ndarray
        Data points in (N, D) matrix.
    y : numpy.ndarray
        Data points in (N, D) matrix.
    V : numpy.ndarray
        Weight matrix (D, D), if `V=None`, `V=eye(D)` is used

    Returns
    -------
    : numpy.ndarray
        Pair-wise Mahalanobis distance of rows of x and y with given weight matrix V.
    """
    if V is None:
        V = np.eye(x.shape[1])
    x2V = np.sum(x.dot(V) * x, 1)
    y2V = np.sum(y.dot(V) * y, 1)
    return (x2V[:, na] + y2V[:, na].T) - 2 * x.dot(V).dot(y.T)


class Kernel(object, metaclass=ABCMeta):

    def __init__(self, dim, par, jitter):
        """
        Kernel base class.

        Parameters
        ----------
        dim : int
            Input dimension
        par : numpy.ndarray
            Kernel parameters in a (dim_out, num_par) matrix, where i-th row contains parameters for i-th output.
        jitter : float
            Jitter for stabilizing inversion of kernel matrix.
        """

        # ensure parameter is 2d array of type float
        self.par = np.atleast_2d(par).astype(float)
        assert self.par.ndim == 2, "Kernel parameters must be 2D array, you donkey!"  # in case ndim > 2
        self.scale = self.par[:, 0]
        self.dim = dim
        self.jitter = jitter
        self.eye_d = np.eye(dim)  # pre-allocation for convenience

    @staticmethod
    def _cho_inv(A, b=None):
        """
        Solution of a linear system :math:`Ax = b`, where :math:`A` is a symmetric positive definite matrix.

        Parameters
        ----------
        A : numpy.ndarray
            Symmetric positive definite matrix.
        b : numpy.ndarray
            Right-hand side. If `b=None` defaults to unit matrix of the same shape as :math:`A`.

        Returns
        -------
        : numpy.ndarray
            If `b=None`, an :math:`A^{-1}` is returned, otherwise :math:`A^{-1}b` is returned.

        """

        b = np.eye(A.shape[0]) if b is None else b

        # solve a system involving  symmetric PD matrix A using Cholesky decomposition
        iA = cho_solve(cho_factor(A), b)

        # inverse of symmetric PD matrix must be symmetric
        iA = 0.5 * (iA + iA.T)
        return iA

    # evaluation
    @abstractmethod
    def eval(self, par, x1, x2=None, diag=False, scaling=True):
        """
        Evaluated kernel for all pair of data points.

        Parameters
        ----------
        par : array_like
        x1 : numpy.ndarray
        x2 : numpy.ndarray
        diag : bool
            If True, return only diagonal of the kernel matrix.
        scaling : bool
            Use kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Kernel matrix of shape `(N, N)`.
        """
        pass

    def eval_inv_dot(self, par, x, b=None, scaling=True):
        # if b=None returns inverse of K
        return Kernel._cho_inv(self.eval(par, x, scaling=scaling) + self.jitter * np.eye(x.shape[1]), b)

    def eval_chol(self, par, x, scaling=True):
        return la.cholesky(self.eval(par, x, scaling=scaling) + self.jitter * np.eye(x.shape[1]))

    def get_parameters(self, par=None):
        if par is None:
            # return parameters kernel was initialized with
            return self.par
        else:

            # ensure supplied kernel parameters are in 2d float array
            par = np.atleast_2d(par).astype(float)
            assert par.ndim == 2, "Supplied Kernel parameters must be a 2d array of shape (dim_out, dim)."

            # returned supplied kernel parameters
            return par

    # expectations
    @abstractmethod
    def exp_x_kx(self, par, x):
        """
        Computes :math:`\mathbb{E}_{x}[k(x, x_i \mid \theta_m)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        par : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Notes
        -----
        Also known as kernel mean map.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\theta_m` returned
            in an array of shape `(N, )`, where `N = x.shape[1]`.
        """
        pass

    @abstractmethod
    def exp_x_xkx(self, par, x):
        """
        Computes :math:`\mathbb{E}_{x}[xk(x, x_i \mid \theta_m)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        par : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\theta_m` returned
            in an array of shape `(D, N)`, where `(D, N) = x.shape`.
        """
        pass

    @abstractmethod
    def exp_x_kxkx(self, par_0, par_1, x):
        """
        Computes :math:`\mathbb{E}_{x}[k(x, x_i \mid \theta_m)k(x, x_j \mid \theta_n)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        par_0 : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.
        par_1 : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Notes
        -----
        Also known as kernel iteration.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i,\ x_j` and vectors of kernel parameters :math:`\theta_m` and
            :math:`\theta_n` returned in an array of shape `(N, N)`, where `N = x.shape[1]`.
        """
        pass

    @abstractmethod
    def exp_x_kxx(self, par):
        """
        Computes :math:`\mathbb{E}_{x}[k(x, x \mid \theta_m)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        par : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\theta_m` returned
            in an array of shape `(N, )`, where `N = x.shape[1]`.
        """
        pass

    @abstractmethod
    def exp_xy_kxy(self, par):
        """
        Computes :math:`\mathbb{E}_{x,x'}[k(x, x' \mid \theta_m)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        par : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for and vector of kernel parameters :math:`\theta_m` returned in an array of shape `(1, )`.
        """
        pass

    # derivatives
    @abstractmethod
    def der_par(self, par_0, x):
        # evaluates derivative of the kernel matrix at par_0; x is data, now acting as parameter
        pass


class RBF(Kernel):

    def __init__(self, dim, par, jitter=1e-8):
        """
        Radial Basis Function kernel.

        .. math::
           k(x, x') = s^2 \exp\left(-\frac{1}{2}(x - x')^{\top}\ Lambda^{-1} (x - x') \right)

        Parameters
        ----------
        dim : int
            Input dimension
        par : numpy.ndarray
            Kernel parameters in a matrix of shape (dim_out, num_par), where i-th row contains parameters for i-th
            output. Each row is :math: `[s, \ell_1, \ldots, \ell_dim]`
        jitter : float
            Jitter for stabilizing inversion of the kernel matrix. Default ``jitter=1e-8``.

        Notes
        -----
        The kernel is also known as Squared Exponential (popular, but wrong), Exponentiated Quadratic (too mouthful)
        or Gaussian (conflicts with terminology for PDFs).
        """

        assert par.shape[1] == dim + 1
        super(RBF, self).__init__(dim, par, jitter)

    def __str__(self):  # TODO: improve string representation
        return '{} {}'.format(self.__class__.__name__, self.par.update({'jitter': self.jitter}))

    def eval(self, par, x1, x2=None, diag=False, scaling=True):
        if x2 is None:
            x2 = x1.copy()

        alpha, sqrt_inv_lam = RBF._unpack_parameters(par)
        alpha = 1.0 if not scaling else alpha

        x1 = sqrt_inv_lam.dot(x1)
        x2 = sqrt_inv_lam.dot(x2)
        if diag:  # only diagonal of kernel matrix
            assert x1.shape == x2.shape
            dx = x1 - x2
            return np.exp(2 * np.log(alpha) - 0.5 * np.sum(dx * dx, axis=0))
        else:
            return np.exp(2 * np.log(alpha) - 0.5 * maha(x1.T, x2.T))

    def exp_x_kx(self, par, x, scaling=False):
        # a.k.a. kernel mean map w.r.t. standard Gaussian PDF
        # par (D+1,) array_like
        alpha, sqrt_inv_lam = RBF._unpack_parameters(par)
        alpha = 1.0 if not scaling else alpha

        inv_lam = sqrt_inv_lam ** 2
        lam = np.diag(inv_lam.diagonal() ** -1)

        c = alpha ** 2 * (la.det(inv_lam + self.eye_d)) ** -0.5
        xl = la.inv(lam + self.eye_d).dot(x)
        return c * np.exp(-0.5 * np.sum(x * xl, axis=0))

    def exp_x_xkx(self, par, x):
        alpha, sqrt_inv_lam = RBF._unpack_parameters(par)
        lam = np.diag(sqrt_inv_lam.diagonal() ** -2)

        mu_q = la.inv(lam + self.eye_d).dot(x)
        q = self.exp_x_kx(par, x)
        return q[na, :] * mu_q

    def exp_x_kxkx(self, par_0, par_1, x, scaling=False):
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
        xi = 2 * np.log(alpha) - 0.5 * np.sum(xi * xi, axis=0)  # (N, )

        # \xi_j^T * \Lambda_n * \xi_j
        xi_1 = sqrt_inv_lam_1.dot(x)  # (D, N)
        xi_1 = 2 * np.log(alpha_1) - 0.5 * np.sum(xi_1 * xi_1, axis=0)  # (N, )

        # \Lambda^{-1} * x
        x_0 = inv_lam.dot(x)  # (D, N)
        x_1 = inv_lam_1.dot(x)

        # R^{-1} = (\Lambda_m^{-1} + \Lambda_n^{-1} + \eye)^{-1}
        r = inv_lam + inv_lam_1 + self.eye_d  # (D, D)

        n = (xi[:, na] + xi_1[na, :]) + 0.5 * maha(x_0.T, -x_1.T, V=la.inv(r))  # (N, N)
        return la.det(r) ** -0.5 * np.exp(n)

    def exp_x_kxx(self, par):
        alpha, sqrt_inv_lam = RBF._unpack_parameters(par)
        return alpha ** 2

    def exp_xy_kxy(self, par):
        alpha, sqrt_inv_lam = RBF._unpack_parameters(par)
        inv_lam = sqrt_inv_lam ** 2
        return alpha ** 2 * la.det(2 * inv_lam + self.eye_d) ** -0.5

    def der_par(self, par_0, x):  # K as kwarg would save computation (would have to be evaluated w/ par_0)
        # par_0: array_like [alpha, el_1, ..., el_D]
        # x: (D, N)
        alpha, el = par_0[0], par_0[1:]
        K = self.eval(par_0, x)
        # derivative w.r.t. alpha (N,N)
        d_alpha = 2 * alpha ** -1 * K
        # derivatives w.r.t. el_1, ..., el_D (N,N,D)
        d_el = (x[:, na, :] - x[:, :, na]) ** 2 * (el ** -3)[:, na, na] * K[na, :, :]
        return np.concatenate((d_alpha[..., na], d_el.T), axis=2)

    @staticmethod
    def _unpack_parameters(par):
        """
        Extract scaling parameter and square-root of inverse lengthscale matrix from vector of kernel parameters.

        Parameters
        ----------
        par : numpy.ndarray

        Returns
        -------
        : tuple

        """
        par = par.astype(float).squeeze()
        # TODO: return scaling and lengthscale, not sqrt inv lambda
        return par[0], np.diag(par[1:] ** -1)


class RBFStudent(RBF):
    """
    RBF kernel with Student's expectations approximated by Monte Carlo.
    """

    def __init__(self, dim, par, jitter=1e-8, dof=4.0, num_mc=1000):

        # samples from standard Student's density
        mean = np.zeros((dim, ))
        cov = np.eye(dim)
        self.num_mc = num_mc
        self.x_samples = self._multivariate_t(mean, cov, dof, size=num_mc).T  # (D, MC)
        super(RBFStudent, self).__init__(dim, par, jitter)

    def exp_x_kx(self, par, x, scaling=False):
        return (1/self.num_mc) * self.eval(par, self.x_samples, x, scaling=scaling).sum(axis=0)

    def exp_x_xkx(self, par, x):
        k = self.eval(par, self.x_samples, x, scaling=False)  # (MC, N)
        return (1 / self.num_mc) * (self.x_samples[..., na] * k[na, ...]).sum(axis=1)

    def exp_x_kxkx(self, par_0, par_1, x, scaling=False):
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

        k0 = self.eval(par_0, self.x_samples, x, scaling=scaling)  # (MC, N)
        k1 = self.eval(par_1, self.x_samples, x, scaling=scaling)
        return (1/self.num_mc) * (k0[:, na, :] * k1[..., na]).sum(axis=0)

    def exp_x_kxx(self, par):
        k = self.eval(par, self.x_samples, self.x_samples, diag=True, scaling=True)
        return (1/self.num_mc) * k.sum()

    def exp_xy_kxy(self, par):
        return (1/self.num_mc) * self.eval(par, self.x_samples, self.x_samples).sum()

    @staticmethod
    def _multivariate_t(mean, scale, nu, size=None):
        """
        Samples of a random variable :math:`X` following a multivariate t-distribution
        :math:`X \sim \mathrm{St}(\mu, \Sigma, \nu)`.

        Parameters
        ----------
        mean
            Mean vector
        scale
            Scale matrix
        nu : float
            Degrees of freedom
        size : int or tuple of ints


        Notes
        -----
        If :math:`y \sim \mathrm{N}(0, \Sigma)` and :math:`u \sim \mathrm{Gamma}(k=\nu/2, \theta=2/\nu)`,
        then :math:`x \sim \mathrm{St}(\mu, \Sigma, \nu)`, where :math:`x = \mu + \frac{y}{\sqrt{u}}`.

        Returns
        -------

        """
        v = np.random.gamma(nu / 2, 2 / nu, size)[:, na]
        n = np.random.multivariate_normal(np.zeros_like(mean), scale, size)
        return mean[na, :] + n / np.sqrt(v)


class RQ(Kernel):

    def __init__(self, dim, par, jitter=1e-8):
        """
        Rational Quadratic kernel.

        .. math::
        \[
           k(x, x') = s^2 \left( 1 + \frac{1}{2\alpha}(x - x')^{\top}\ Lambda^{-1} (x - x') \right)^{-\alpha}
        \]

        Parameters
        ----------
        dim : int
            Input dimension
        par : numpy.ndarray
            Kernel parameters in a matrix of shape (dim_out, num_par), where i-th row contains parameters
            for i-th output. Each row is :math: `[s, \alpha, \ell_1, \ldots, \ell_dim]`
        jitter : float
            Jitter for stabilizing inversion of the kernel matrix. Default ``jitter=1e-8``.

        Notes
        -----
        The kernel expectations are w.r.t standard Student's t density and are approximate.
        """
        assert par.shape[1] == dim + 2
        super(RQ, self).__init__(dim, par, jitter)

    def eval(self, par, x1, x2=None, diag=False, scaling=True):
        if x2 is None:
            x2 = x1.copy()

        s, alpha, sqrt_inv_lam = RQ._unpack_parameters(par)
        s = 1.0 if not scaling else s

        x1 = sqrt_inv_lam.dot(x1)
        x2 = sqrt_inv_lam.dot(x2)
        if diag:  # diagonal only
            assert x1.shape == x2.shape
            dx = x1 - x2
            return s**2 * (1 + (2*alpha) ** -1 * np.sum(dx * dx, axis=0)) ** (-alpha)
        else:
            return s**2 * (1 + (2*alpha) ** -1 * maha(x1.T, x2.T)) ** (-alpha)

    def exp_x_kx(self, par, x, scaling=False):
        """
        RQ kernel mean

        .. math::
        \[
            \mathbb{E}_{x}{k(x, x_i)} = \int\! k(x, x_i) St(x \mid 0, I, \nu) \,\mathrm{d}x
        \]

        Parameters
        ----------
        par : numpy.ndarray
        x : numpy.ndarray

        Returns
        -------

        """
        s, alpha, sqrt_inv_lam = RQ._unpack_parameters(par)
        s = 1.0 if not scaling else s

        inv_lam = sqrt_inv_lam ** 2
        lam = np.diag(inv_lam.diagonal() ** -1)

        c = s ** 2 * la.det(inv_lam + self.eye_d) ** -0.5
        xl = la.inv(lam + self.eye_d).dot(x)
        return c * (1 + (2*alpha)**-1 * np.sum(x * xl, axis=0)) ** (-alpha)

    def exp_x_xkx(self, par, x):
        """
        RQ kernel cross-correlation

        .. math::
        \[
            \mathbb{E}_{x}[xk(x, x_i)] = \int\! xk(x, x_i) St(x \mid 0, I, \nu) \mathrm{d}x
        \]

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        par : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\theta_m`
            returned
            in an array of shape `(D, N)`, where `(D, N) = x.shape`.
        """
        s, alpha, sqrt_inv_lam = RQ._unpack_parameters(par)
        lam = np.diag(sqrt_inv_lam.diagonal() ** -2)

        mu_q = la.inv(lam + self.eye_d).dot(x)
        q = self.exp_x_kx(par, x)
        return q[na, :] * mu_q

    def exp_x_kxkx(self, par_0, par_1, x, scaling=False):
        """
        RQ kernel correlation

        .. math:
        \[
            \mathbb{E}[k(x, x_i), k(x, x_j)] = \int\! k(x, x_i), k(x, x_j) St(x \mid 0, I, \nu)\, \mathrm{d}x
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
        s, alpha, sqrt_inv_lam = RQ._unpack_parameters(par_0)
        s_1, alpha_1, sqrt_inv_lam_1 = RQ._unpack_parameters(par_1)
        s, s_1 = (1.0, 1.0) if not scaling else (s, s_1)
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
        r = inv_lam + inv_lam_1 + self.eye_d  # (D, D)

        n = (xi[:, na] + xi_1[na, :]) + maha(x_0.T, -x_1.T, V=la.inv(r))  # (N, N)
        return s**2 * s_1**2 * la.det(r) ** -0.5 * (1 + (2*alpha) ** -1 * n) ** (-alpha)

    def exp_x_kxx(self, par):
        return par[0] ** 2

    def exp_xy_kxy(self, par):
        s, alpha, sqrt_inv_lam = RQ._unpack_parameters(par)
        inv_lam = sqrt_inv_lam ** 2
        return s ** 2 * la.det(2 * inv_lam + self.eye_d) ** -0.5

    def der_par(self, par_0, x):
        pass

    @staticmethod
    def _unpack_parameters(par):
        """
        Break down the parameter vector into individual parameters.

        Parameters
        ----------
        par : numpy.ndarray

        Returns
        -------
        : tuple

        """
        par = par.astype(float).squeeze()
        return par[0], par[1], np.diag(par[2:] ** -1)
