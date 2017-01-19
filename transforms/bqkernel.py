from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve


# TODO: documentation


class Kernel(object, metaclass=ABCMeta):

    def __init__(self, dim, dim_out, hypers, jitter):
        """
        Kernel base class.

        Parameters
        ----------
        dim : int
            Input dimension
        dim_out : int
            Output dimension
        hypers : numpy.ndarray
            Kernel parameters in a (dim_out, num_hyp) matrix, where i-th row contains parameters for i-th output.
        jitter : float
            Jitter for stabilizing inversion of kernel matrix.
        """

        # ensure parameter matrix is 2d array
        self.hypers = np.atleast_2d(hypers).astype(float)
        assert self.hypers.ndim == 2  # in case ndim > 2
        assert hypers.shape[0] == dim_out

        # inputs, # outputs and # params per output
        self.dim = dim
        self.dim_out, self.num_hyp = self.hypers.shape
        self.jitter = jitter

        # identity matrices for convenience
        self.eye_d = np.eye(dim)

    @staticmethod
    def _cho_inv(A, b=None):
        # inversion of PD matrix A using Cholesky decomposition
        if b is None:
            b = np.eye(A.shape[0])
        return cho_solve(cho_factor(A), b)

    # evaluation
    @abstractmethod
    def eval(self, x1, x2=None, hyp=None, diag=False):
        pass

    def eval_inv_dot(self, x, hyp=None, b=None, ignore_alpha=False):
        # if b=None returns inverse of K
        return Kernel._cho_inv(self.eval(x, hyp=hyp, ignore_alpha=ignore_alpha) + self.jitter * np.eye(x.shape[1]), b)

    def eval_chol(self, x, hyp=None, ignore_alpha=False):
        return la.cholesky(self.eval(x, hyp=hyp, ignore_alpha=ignore_alpha) + self.jitter * np.eye(x.shape[1]))

    # kernel mean, "covariance", "cross-covariance"
    def mean(self, x):
        n = x.shape[1]
        # FIXME: no need to for the condition, the code inside works for dim_out=1 as well
        if self.dim_out > 1:
            q = np.zeros((n, self.dim_out))
            for i in range(self.dim_out):
                q[:, i] = self.exp_x_kx(x, self.hypers[i, :])
            return q
        else:
            return self.exp_x_kx(x, self.hypers[0, :])

    def covariance(self, x):
        n = x.shape[1]
        if self.dim_out > 1:
            Q = np.zeros((n, n, self.dim_out, self.dim_out))
            for i in range(self.dim_out):
                for j in range(self.dim_out):
                    Q[..., i, j] = self.exp_x_kxkx(x, self.hypers[i, :], self.hypers[j, :])
            return Q
        else:
            return self.exp_x_kxkx(x, self.hypers[0, :], self.hypers[0, :])

    def crosscovariance(self, x):
        n = x.shape[1]
        if self.dim_out > 1:
            R = np.zeros((self.dim, n, self.dim_out))
            for i in range(self.dim_out):
                    R[..., i] = self.exp_x_xkx(x, self.hypers[i, :])
            return R
        else:
            return self.exp_x_xkx(x, self.hypers[0, :])

    # expectations
    @abstractmethod
    def exp_x_kx(self, x, hyp):
        """
        Computes

        .. math::
           \mathbb{E}_{x}[k(x, x_i \mid \theta_m)]

        where `i` is datapoint index and `m` is parameter index.

        Parameters
        ----------
        x
        hyp

        Returns
        -------

        """
        pass

    @abstractmethod
    def exp_x_xkx(self, x, hyp):
        """
        Computes

        .. math::
           \mathbb{E}_{x}[xk(x, x_i \mid \theta_m)]

        where `i` is datapoint index and `m` is parameter index.

        Parameters
        ----------
        x
        hyp

        Returns
        -------

        """
        pass

    @abstractmethod
    def exp_x_kxx(self, hyp):
        """
        Computes

        .. math::
           \mathbb{E}_{x}[k(x, x \mid \theta_m)]

        where `m` is parameter index.

        Parameters
        ----------
        hyp

        Returns
        -------

        """
        pass

    @abstractmethod
    def exp_xy_kxy(self, hyp):
        """
        Computes

        .. math::
           \mathbb{E}_{x,x'}[k(x, x' \mid \theta_m)]

        where `m` is parameter index.

        Parameters
        ----------
        hyp

        Returns
        -------

        """
        pass

    @abstractmethod
    def exp_x_kxkx(self, x, hyp, hyp_1):
        """
        Computes

        .. math::
           \mathbb{E}_{x}[k(x, x_i \mid \theta_m)k(x, x_j \mid \theta_n)]

        where `i` and `j` are datapoint indexes and `m` and `n` are parameter indexes.

        Parameters
        ----------
        x
        hyp
        hyp_1

        Returns
        -------

        """
        pass

    @abstractmethod
    def exp_model_variance(self, x, hyp):
        # FIXME: model dependent should be in Model
        pass

    @abstractmethod
    def integral_variance(self, x, hyp):
        # FIXME: model dependent should be in Model
        pass

    # derivatives
    @abstractmethod
    def der_hyp(self, x, hyp0):
        # evaluates derivative of the kernel matrix at hyp0; x is data, now acting as parameter
        pass

    @abstractmethod
    def _get_hyperparameters(self, hyp):
        pass


class RBF(Kernel):
    # _hyperparameters_ = ['alpha', 'el']

    def __init__(self, dim, dim_out, hypers, jitter):
        """
        Radial Basis Function kernel.

        Parameters
        ----------
        dim : int
            Input dimension
        dim_out : int
            Output dimension
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

        assert hypers.shape == (dim_out, dim + 1)
        super(RBF, self).__init__(dim, dim_out, hypers, jitter)

    def __str__(self):  # TODO: improve string representation
        return '{} {}'.format(self.__class__.__name__, self.hypers.update({'jitter': self.jitter}))

    def eval(self, x1, x2=None, hyp=None, diag=False, ignore_alpha=False):
        # x1.shape = (D, N), x2.shape = (D, M), hyp (D+1,) array_like
        if x2 is None:
            x2 = x1
        # use hyp as hypers if given, otherwise use init hypers
        alpha, sqrt_inv_lam = RBF._unpack_parameters(hyp)
        alpha = 1.0 if ignore_alpha else alpha

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

    def exp_model_variance(self, x, hyp):
        alpha, sqrt_inv_lam = self._get_hyperparameters(hyp)
        Q = self.exp_x_kxkx(x, hyp=hyp)
        iK = self.eval_inv_dot(x, hyp=hyp, ignore_alpha=True)
        return alpha**2 * (1 - np.trace(Q.dot(iK)))

    def integral_variance(self, x, hyp):
        alpha, sqrt_inv_lam = self._get_hyperparameters(hyp)
        q = self.exp_x_kx(x, hyp)
        iK = self.eval_inv_dot(x, hyp=hyp, ignore_alpha=True)
        return alpha**2 * (la.det(2 * sqrt_inv_lam ** 2 + self.eye_d) ** -0.5 - q.T.dot(iK).dot(q))

    def der_hyp(self, x, hyp0):  # K as kwarg would save computation (would have to be evaluated w/ hyp0)
        # hyp0: array_like [alpha, el_1, ..., el_D]
        # x: (D, N)
        alpha, el = hyp0[0], hyp0[1:]
        K = self.eval(x, hyp=hyp0)
        # derivative w.r.t. alpha (N,N)
        d_alpha = 2 * alpha ** -1 * K
        # derivatives w.r.t. el_1, ..., el_D (N,N,D)
        d_el = (x[:, na, :] - x[:, :, na]) ** 2 * (el ** -3)[:, na, na] * K[na, :, :]
        return np.concatenate((d_alpha[..., na], d_el.T), axis=2)

    @staticmethod
    def _unpack_parameters(param):
        # turn vector of kernel parameters into variables
        return param[0], np.diag(param[1:] ** -1)

    def _get_hyperparameters(self, hyp=None):
        # if new hypers are given return them, if not return the initial hypers
        if hyp is None:
            return self.alpha, self.sqrt_inv_lam
        else:
            hyp = np.asarray(hyp, dtype=float)
            return hyp[0], np.diag(hyp[1:] ** -1)

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


class Affine(Kernel):
    def __init__(self, dim, dim_out, hypers):
        super(Affine, self).__init__(dim, 1, hypers)

    def eval(self, x1, x2=None, hyp=None, diag=False):
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

    def _get_default_hyperparameters(self, dim):
        pass
