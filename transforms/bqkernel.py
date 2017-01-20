from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve


# TODO: documentation


class Kernel(object, metaclass=ABCMeta):

    def __init__(self, dim, hypers, jitter):
        """
        Kernel base class.

        Parameters
        ----------
        dim : int
            Input dimension
        hypers : numpy.ndarray
            Kernel parameters in a (dim_out, num_hyp) matrix, where i-th row contains parameters for i-th output.
        jitter : float
            Jitter for stabilizing inversion of kernel matrix.
        """

        # ensure parameter is 2d array of type float
        self.hypers = np.atleast_2d(hypers).astype(float)
        assert self.hypers.ndim == 2, "Kernel parameters must be 2D array, you donkey!"  # in case ndim > 2
        self.scale = self.hypers[:, 0]
        self.dim = dim
        self.jitter = jitter
        self.eye_d = np.eye(dim)  # pre-allocation for convenience

    @staticmethod
    def _cho_inv(A, b=None):
        """

        Parameters
        ----------
        A : numpy.ndarray
        b : numpy.ndarray

        Returns
        -------
        : numpy.ndarray

        """
        # inversion of PD matrix A using Cholesky decomposition
        if b is None:
            b = np.eye(A.shape[0])
        return cho_solve(cho_factor(A), b)

    # evaluation
    @abstractmethod
    def eval(self, hyp, x1, x2=None, diag=False, scaling=True):
        """
        Evaluated kernel for all pair of data points.

        Parameters
        ----------
        hyp : array_like
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

    def eval_inv_dot(self, hyp, x, b=None, scaling=True):
        # if b=None returns inverse of K
        return Kernel._cho_inv(self.eval(hyp, x, scaling=scaling) + self.jitter * np.eye(x.shape[1]), b)

    def eval_chol(self, hyp, x, scaling=True):
        return la.cholesky(self.eval(hyp, x, scaling=scaling) + self.jitter * np.eye(x.shape[1]))

    # expectations
    @abstractmethod
    def exp_x_kx(self, x, hyp):
        """
        Computes :math:`\mathbb{E}_{x}[k(x, x_i \mid \theta_m)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        hyp : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\theta_m` returned
            in an array of shape `(N, )`, where `N = x.shape[1]`.
        """
        pass

    @abstractmethod
    def exp_x_xkx(self, x, hyp):
        """
        Computes :math:`\mathbb{E}_{x}[xk(x, x_i \mid \theta_m)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        hyp : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\theta_m` returned
            in an array of shape `(D, N)`, where `(D, N) = x.shape`.
        """
        pass

    @abstractmethod
    def exp_x_kxx(self, hyp):
        """
        Computes :math:`\mathbb{E}_{x}[k(x, x \mid \theta_m)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        hyp : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\theta_m` returned
            in an array of shape `(N, )`, where `N = x.shape[1]`.
        """
        pass

    @abstractmethod
    def exp_xy_kxy(self, hyp):
        """
        Computes :math:`\mathbb{E}_{x,x'}[k(x, x' \mid \theta_m)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        hyp : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for and vector of kernel parameters :math:`\theta_m` returned in an array of shape `(1, )`.
        """
        pass

    @abstractmethod
    def exp_x_kxkx(self, x, hyp, hyp_1):
        """
        Computes :math:`\mathbb{E}_{x}[k(x, x_i \mid \theta_m)k(x, x_j \mid \theta_n)]`.

        Parameters
        ----------
        x : numpy.ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).
        hyp : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.
        hyp : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : numpy.ndarray
            Expectation for given data points :math:`x_i,\ x_j` and vectors of kernel parameters :math:`\theta_m` and
            :math:`\theta_n` returned in an array of shape `(N, N)`, where `N = x.shape[1]`.
        """
        pass

    # derivatives
    @abstractmethod
    def der_hyp(self, x, hyp0):
        # evaluates derivative of the kernel matrix at hyp0; x is data, now acting as parameter
        pass

    @abstractmethod
    def get_hyperparameters(self, hyp):
        pass


class RBF(Kernel):

    def __init__(self, dim, hypers, jitter=1e-8):
        """
        Radial Basis Function kernel.

        .. math::
           k(x, x') = s^2 \exp(-\frac{1}{2}(x - x')^{\top}\ Lambda^{-1} (x - x') \right)

        Parameters
        ----------
        dim : int
            Input dimension
        hypers : numpy.ndarray
            Kernel parameters in a matrix of shape (dim_out, num_hyp), where i-th row contains parameters for i-th
            output. Each row is :math: `[\alpha, \ell_1, \ldots, \ell_dim]`
        jitter : float
            Jitter for stabilizing inversion of the kernel matrix. Default ``jitter=1e-8``.

        Notes
        -----
        The kernel is also known as Squared Exponential (popular, but wrong), Exponentiated Quadratic (too mouthful)
        or Gaussian (conflicts with terminology for PDFs).
        """

        assert hypers.shape[1] == dim + 1
        super(RBF, self).__init__(dim, hypers, jitter)

    def __str__(self):  # TODO: improve string representation
        return '{} {}'.format(self.__class__.__name__, self.hypers.update({'jitter': self.jitter}))

    def eval(self, hyp, x1, x2=None, diag=False, scaling=True):
        if x2 is None:
            x2 = x1.copy()

        alpha, sqrt_inv_lam = RBF._unpack_parameters(hyp)
        alpha = 1.0 if scaling else alpha

        x1 = sqrt_inv_lam.dot(x1)
        x2 = sqrt_inv_lam.dot(x2)
        if diag:  # only diagonal of kernel matrix
            assert x1.shape == x2.shape
            dx = x1 - x2
            return np.exp(2 * np.log(alpha) - 0.5 * np.sum(dx * dx, axis=0))
        else:
            return np.exp(2 * np.log(alpha) - 0.5 * self._maha(x1.T, x2.T))

    def exp_x_kx(self, x, hyp, ignore_alpha=True):
        # a.k.a. kernel mean map w.r.t. standard Gaussian PDF
        # hyp (D+1,) array_like
        alpha, sqrt_inv_lam = RBF._unpack_parameters(hyp)
        alpha = 1.0 if ignore_alpha else alpha

        inv_lam = sqrt_inv_lam ** 2
        lam = np.diag(inv_lam.diagonal() ** -1)

        c = alpha ** 2 * (la.det(inv_lam + self.eye_d)) ** -0.5
        xl = la.inv(lam + self.eye_d).dot(x)
        return c * np.exp(-0.5 * np.sum(x * xl, axis=0))

    def exp_x_xkx(self, x, hyp):
        alpha, sqrt_inv_lam = RBF._unpack_parameters(hyp)
        lam = np.diag(sqrt_inv_lam.diagonal() ** -2)

        mu_q = la.inv(lam + self.eye_d).dot(x)
        q = self.exp_x_kx(x, hyp)
        return q[na, :] * mu_q

    def exp_x_kxx(self, hyp):
        alpha, sqrt_inv_lam = RBF._unpack_parameters(hyp)
        return alpha ** 2

    def exp_xy_kxy(self, hyp):
        alpha, sqrt_inv_lam = RBF._unpack_parameters(hyp)
        inv_lam = sqrt_inv_lam ** 2
        return alpha ** 2 * la.det(2 * inv_lam + self.eye_d) ** -0.5

    def exp_x_kxkx(self, x, hyp, hyp_1, ignore_alpha=True):
        """
        "Correlation" matrix of kernels with elements

        .. math:
        \[
            \mathbb{E}[k(x, x_i), k(x, x_j)]
        \]

        Parameters
        ----------
        x : numpy.ndarray of shape (D, N)
        hyp : numpy.ndarray of shape (D, )
        hyp_1 : numpy.ndarray of shape (D, )
        ignore_alpha : bool


        Returns
        -------

        """
        alpha, sqrt_inv_lam = RBF._unpack_parameters(hyp)
        alpha_1, sqrt_inv_lam_1 = RBF._unpack_parameters(hyp_1)
        alpha, alpha_1 = (1.0, 1.0) if ignore_alpha else alpha, alpha_1
        inv_lam = sqrt_inv_lam ** 2
        inv_lam_1 = sqrt_inv_lam_1 ** 2

        # \xi_i^T * \Lambda_m * \xi_i
        xi = sqrt_inv_lam.dot(x)  # (D, N)
        xi = 2 * np.log(alpha) - 0.5 * np.sum(xi * xi, axis=0)  # (N, )

        # \xi_j^T * \Lambda_n * \xi_j
        xi_1 = sqrt_inv_lam_1.dot(x)  # (D, N)
        xi_1 = 2 * np.log(alpha_1) - 0.5 * np.sum(xi_1 * xi_1, axis=0)  # (N, )

        # \Lambda^{-1} * x
        x = inv_lam.dot(x)  # (D, N)
        x_1 = inv_lam_1.dot(x)

        # R^{-1} = (\Lambda_m^{-1} + \Lambda_n^{-1} + \eye)^{-1}
        r = inv_lam + inv_lam_1 + self.eye_d  # (D, D)

        n = (xi[:, na] + xi_1[na, :]) + 0.5 * self._maha(x.T, -x_1.T, V=la.inv(r))  # (N, N)
        return la.det(r) ** -0.5 * np.exp(n)

    def der_hyp(self, x, hyp0):  # K as kwarg would save computation (would have to be evaluated w/ hyp0)
        # hyp0: array_like [alpha, el_1, ..., el_D]
        # x: (D, N)
        alpha, el = hyp0[0], hyp0[1:]
        K = self.eval(hyp=hyp0, x1=hyp0)
        # derivative w.r.t. alpha (N,N)
        d_alpha = 2 * alpha ** -1 * K
        # derivatives w.r.t. el_1, ..., el_D (N,N,D)
        d_el = (x[:, na, :] - x[:, :, na]) ** 2 * (el ** -3)[:, na, na] * K[na, :, :]
        return np.concatenate((d_alpha[..., na], d_el.T), axis=2)

    @staticmethod
    def _unpack_parameters(param):
        # divide kernel parameters into kernel scaling and square-root of inverse lengthscale matrix
        return param[0], np.diag(param[1:] ** -1)

    def get_hyperparameters(self, hyp=None):
        if hyp is None:
            # return parameters kernel was initialized with
            return self.hypers
        else:

            # ensure supplied kernel parameters are in 2d float array
            hyp = np.asarray(hyp, dtype=float)
            assert hyp.ndim == 2, "Supplied Kernel parameters must be a 2d array of shape (dim_out, dim)."

            # returned supplied kernel parameters
            return hyp

    def _maha(self, x, y, V=None):
        """
        Pair-wise Mahalanobis distance of rows of x and y with given weight matrix V.
        :param x: (n, d) matrix of row vectors
        :param y: (n, d) matrix of row vectors
        :param V: weight matrix (d, d), if V=None, V=eye(d) is used
        :return:
        """
        if V is None:
            V = np.eye(x.shape[1])
        x2V = np.sum(x.dot(V) * x, 1)
        y2V = np.sum(y.dot(V) * y, 1)
        return (x2V[:, na] + y2V[:, na].T) - 2 * x.dot(V).dot(y.T)


class RQ(Kernel):

    def __init__(self, dim, hypers, jitter=1e-8):
        """
        Rational Quadratic kernel.

        .. math::
           k(x, x') = s^2 \left( 1 + \frac{1}{2\alpha}(x - x')^{\top}\ Lambda^{-1} (x - x') \right)^{-\alpha}

        Parameters
        ----------
        dim : int
            Input dimension
        hypers : numpy.ndarray
            Kernel parameters in a matrix of shape (dim_out, num_hyp), where i-th row contains parameters
            for i-th
            output. Each row is :math: `[\alpha, \ell_1, \ldots, \ell_dim]`
        jitter : float
            Jitter for stabilizing inversion of the kernel matrix. Default ``jitter=1e-8``.

        Notes
        -----
        The kernel expectations are w.r.t standard Student's t density and are approximate.
        """
        assert hypers.shape[1] == dim+2
        super(RQ, self).__init__(dim, hypers, jitter)

    def eval(self, hyp, x1, x2=None, diag=False, scaling=True):
        pass

    def exp_x_kx(self, x, hyp=None):
        pass

    def exp_x_xkx(self, x, hyp=None):
        pass

    def exp_x_kxx(self, hyp=None):
        pass

    def exp_xy_kxy(self, hyp=None):
        pass

    def exp_x_kxkx(self, x, hyp=None, hyp_1=None):
        pass

    def der_hyp(self, x, hyp0):
        pass

    def get_hyperparameters(self, hyp):
        pass