from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve
from scipy.special import factorial, factorial2

from ssmtoybox.utils import maha, multivariate_t


class Kernel(object, metaclass=ABCMeta):
    """
    Kernel base class.

    Parameters
    ----------
    dim : int
        Input dimension.

    par : (dim_out, num_par) ndarray
        Kernel parameters in a matrix, where i-th row contains parameters for i-th output.

    jitter : float
        Jitter for stabilizing inversion of kernel matrix.
    """

    def __init__(self, dim, par, jitter):
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
        A : ndarray
            Symmetric positive definite matrix.

        b : None or ndarray
            Right-hand side. If `b=None` defaults to unit matrix of the same shape as `A`.

        Returns
        -------
        : ndarray
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
        par : ndarray
            Kernel parameters.

        x1 : ndarray
            Data set.

        x2 : ndarray, optional
            Data set. If `None`, correlations between `x1`, `x1` are computed.

        diag : bool, optional
            If `True`, return only diagonal of the kernel matrix.

        scaling : bool, optional
            Use kernel scaling parameter.

        Returns
        -------
        : ndarray
            Kernel matrix of shape `(N, N)`.
        """
        pass

    def eval_inv_dot(self, par, x, b=None, scaling=True):
        """
        Compute the product of kernel matrix inverse and a vector `b`.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        x : ndarray
            Data set.

        b : None or ndarray, optional
            If `None`, inverse kernel matrix is computed (i.e. `b=np.eye(N)`).

        scaling : bool, optional
            Use scaling parameter of the kernel matrix.

        Returns
        -------
        : (N, N) ndarray
            Product of kernel matrix inverse and vector `b`.
        """
        # if b=None returns inverse of K
        return Kernel._cho_inv(self.eval(par, x, scaling=scaling) + self.jitter * np.eye(x.shape[1]), b)

    def eval_chol(self, par, x, scaling=True):
        """
        Compute of Cholesky factor of the kernel matrix.

        Parameters
        ----------
        par : (dim+1, ) ndarray
            Kernel parameters.

        x : (dim, N) ndarray
            Data set.

        scaling : bool, optional
            Use scaling parameter of the kernel.

        Returns
        -------
        : (N, N) ndarray
            Cholesky factor of the kernel matrix.
        """
        return la.cholesky(self.eval(par, x, scaling=scaling) + self.jitter * np.eye(x.shape[1]))

    def get_parameters(self, par=None):
        """Get kernel parameters."""
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
        Computes :math:`\\mathbb{E}_{x}[k(x, x_i \\mid \\theta_m)]`.

        Parameters
        ----------
        x : (dim, N) ndarray
            Data (sigma-points).

        par : ndarray
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Notes
        -----
        Also known as kernel mean map.

        Returns
        -------
        : ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\\theta_m` returned
            in an array of shape `(N, )`, where `N = x.shape[1]`.
        """
        pass

    @abstractmethod
    def exp_x_xkx(self, par, x):
        """
        Computes :math:`\\mathbb{E}_{x}[xk(x, x_i \\mid \\theta_m)]`.

        Parameters
        ----------
        x : (dim, N) ndarray
            Data (sigma-points).

        par : ndarray
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\\theta_m` returned
            in an array of shape `(D, N)`, where `(D, N) = x.shape`.
        """
        pass

    @abstractmethod
    def exp_x_kxkx(self, par_0, par_1, x):
        """
        Computes :math:`\\mathbb{E}_{x}[k(x, x_i \\mid \\theta_m)k(x, x_j \\mid \\theta_n)]`.

        Parameters
        ----------
        x : (dim, N) ndarray
            Data (sigma-points).

        par_0 : ndarray
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        par_1 : ndarray
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Notes
        -----
        Also known as kernel iteration.

        Returns
        -------
        : ndarray
            Expectation for given data points :math:`x_i,\\ x_j` and vectors of kernel parameters :math:`\\theta_m` and
            :math:`\\theta_n` returned in an array of shape `(N, N)`, where `N = x.shape[1]`.
        """
        pass

    @abstractmethod
    def exp_x_kxx(self, par):
        """
        Computes :math:`\\mathbb{E}_{x}[k(x, x \\mid \\theta_m)]`.

        Parameters
        ----------
        x : (dim, N) ndarray
            Data (sigma-points).

        par : ndarray
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\\theta_m` returned
            in an array of shape `(N, )`, where `N = x.shape[1]`.
        """
        pass

    @abstractmethod
    def exp_xy_kxy(self, par):
        """
        Computes :math:`\\mathbb{E}_{x,x'}[k(x, x' \\mid \\theta_m)]`.

        Parameters
        ----------
        x : (dim, N) ndarray
            Sigma-points (data).
        par : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : ndarray
            Expectation for and vector of kernel parameters :math:`\\theta_m` returned in an array of shape `(1, )`.
        """
        pass

    # derivatives
    @abstractmethod
    def der_par(self, par_0, x):
        """
        Evaluates derivative of the kernel matrix w.r.t. its parameters at `par_0`.

        Parameters
        ----------
        par_0 : ndarray
            Values of kernel parameters where to evaluate.

        x : ndarray
            Data.

        Returns
        -------
        : ndarray
            Kernel matrix derivatives evaluated at `par_0`.
        """
        pass


class RBF(Kernel):
    """
    Radial Basis Function kernel.

    .. math::
       k(x, x') = s^2 \\exp\\left(-\\frac{1}{2}(x - x')^{\\top}\\Lambda^{-1} (x - x') \\right)

    Parameters
    ----------
    dim : int
        Input dimension.

    par : ndarray
        Kernel parameters in a matrix of shape (dim_out, num_par), where i-th row contains parameters for i-th
        output. Each row is :math:`[s, \\ell_1, \\ldots, \\ell_D]`.

    jitter : float
        Jitter for stabilizing inversion of the kernel matrix. Default ``jitter=1e-8``.

    Notes
    -----
    The kernel is also known as Squared Exponential (popular, but wrong), Exponentiated Quadratic (too mouthful)
    or Gaussian (conflicts with terminology for PDFs).
    """

    def __init__(self, dim, par, jitter=1e-8):
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

        .. math::
            \\mathbb{E}[k(x, x_i), k(x, x_j)] = \\int\\! k(x, x_i), k(x, x_j) N(x \\mid 0, I)\\, \\mathrm{d}x


        Parameters
        ----------
        x : (dim, N) ndarray
            Data points.

        par_0 : (dim, ) ndarray
        par_1 : (dim, ) ndarray
            Kernel parameters.

        scaling : bool, optional
            Kernel scaling parameter used when `scaling=True`.

        Returns
        -------
        : ndarray
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

    def exp_x_px(self, multi_ind):
        """
        Compute expectation \\mathbb{E}[p(x)^T]_{q} for all :math`q`. The expectation is equal to

        .. math::
             \\prod_{d=1}^D (\\alpha_d^q - 1)!!

        when :math:`\\alpha^q_d` is even :math:`\\forall q`. Otherwise the expectation is zero.

        Parameters
        ----------
        multi_ind : (D, Q) ndarray
            Matrix of multi-indices. Each column is a multi-index :math:`\\alpha^q \\in \\mathbb{N}_0^D` defining one
            of the Q multivariate polynomial basis functions.

        Returns
        -------
        : (Q, ) ndarray
            Vector of expectations.
        """
        dim, num_basis = multi_ind.shape
        alpha = multi_ind - 1
        result = np.zeros((num_basis, ))
        for q in range(num_basis):
            all_even = np.all(multi_ind[:, q] % 2 == 0)
            if all_even:
                result[q] = np.prod([factorial2(alpha[d, q], exact=True) for d in range(dim)])
        return result

    def exp_x_xpx(self, multi_ind):
        """
        Compute expectation \\mathbb{E}[xp(x)^T]_{eq} for all :math:`e` and :math`q`. The expectation is equal to

        .. math::
             \\alpha^q_e\\prod_{d \neq e} (\\alpha^q_d - 1)!!

        when :math:`\\alpha^q_e + 1` is even and :math:`\\alpha^q_d, \\forall d \neq e` are even.
        Otherwise the expectation is zero.

        Parameters
        ----------
        multi_ind : (D, Q) ndarray
            Matrix of multi-indices. Each column is a multi-index :math:`\\alpha^q \\in \\mathbb{N}_0^D` defining one
            of the Q multivariate polynomial basis functions.

        Returns
        -------
        : (D, Q) ndarray
            Matrix of expectations.
        """
        dim, num_bases = multi_ind.shape
        d_ind = np.arange(dim)
        result = np.zeros(multi_ind.shape)
        for d in range(dim):
            for q in range(num_bases):
                # all remaining multi-indices even? # i.e. none are odd?
                alpha_min_d = multi_ind[d_ind != d, q]
                all_even = np.all(alpha_min_d % 2 == 0)
                if (multi_ind[d, q] + 1) % 2 == 0 and all_even:
                    amd_fact2 = [factorial2(amd - 1, exact=True) for amd in alpha_min_d]
                    result[d, q] = multi_ind[d, q]*np.prod(amd_fact2)
                else:
                    result[d, q] = 0
        return result

    def exp_x_pxpx(self, multi_ind):
        """
        Compute expectation \\mathbb{E}[p(x)p(x)^T]_{rq} for all :math:`r` and :math`q`. The expectation is equal to

        .. math::
             \\prod_{d = 1}^D (\\alpha^q_d + \\alpha^r_d - 1)!!

        when :math:`\\forall d,\\quad \\alpha^q_d + \\alpha^r_d` are even (where :math:`r` and :math:`q` are fixed).
        Otherwise the expectation is zero.

        Parameters
        ----------
        multi_ind : (D, Q) ndarray
            Matrix of multi-indices. Each column is a multi-index :math:`\\alpha^q \\in \\mathbb{N}_0^D` defining one
            of the Q multivariate polynomial basis functions.

        Returns
        -------
        : (Q, Q) ndarray
            Matrix of expectations.
        """
        dim, num_bases = multi_ind.shape
        result = np.zeros((num_bases, num_bases))
        for r in range(num_bases):
            for q in range(num_bases):
                all_even = np.all((multi_ind[:, r] + multi_ind[:, q]) % 2 == 0)
                if all_even:
                    apa_fact2 = [factorial2(multi_ind[d, r] + multi_ind[d, q] - 1, exact=True) for d in range(dim)]
                    result[r, q] = np.prod(apa_fact2)
                else:
                    result[r, q] = 0
        return result

    def exp_x_kxpx(self, par, multi_ind, x):
        """
        Compute expectation :math:`\\mathbb{E}[k(x)p(x)^T]_{nq}`. For given :math:`n` and :math`q`, the expectation is
        given by

        .. math::
            \\prod_{d=1}^D\left[(1+\ell^2_d)^{\\alpha_{dj}}\\exp\left(-\\frac{x_d^2}{2(1+\ell_d^2)}\right)b_{ijd}\right]

        where

        .. math::
            b_{ijd} = \\sum_{m=0}^{\left\lfloor \\alpha_{dj}/2 \right\rfloor}
            \\frac{\\alpha_{dj}!}{2^j j! (\\alpha_{dj} - 2m)!} \\ell_d^{4m}x_{di}^{\\alpha_{dj}-2m}

        Parameters
        ----------
        par : (dim, ) ndarray
            Kernel parameters.

        multi_ind : (D, Q) ndarray
            Matrix of multi-indices. Each column is a multi-index :math:`\\alpha^q \\in \\mathbb{N}_0^D` defining one
            of the Q multivariate polynomial basis functions.

        x : (dim, N) ndarray
            Data points.

        Returns
        -------
        : (N, Q) ndarray
            Matrix of expectations.
        """
        dim, num_bases = multi_ind.shape
        num_pts = x.shape[1]
        scale, sqrt_inv_lam = RBF._unpack_parameters(par)
        ell = sqrt_inv_lam ** -2
        result = np.zeros((num_pts, num_bases))
        dim_zeros = np.zeros((dim, ))

        fact = lambda num: factorial(num, exact=True)
        for n in range(num_pts):
            for q in range(num_bases):

                # compute each factor in the product
                temp = dim_zeros.copy()
                for d in range(dim):
                    alpha = multi_ind[d, q]
                    # exponential part
                    a = (1 + ell[0, d]**2)**alpha * np.exp(-x[d, n]**2 / (2*(1 + ell[0, d]**2)))

                    # binomial part
                    b = 0
                    for m in range(int(np.floor(alpha/2))+1):
                        part_1 = (fact(alpha) / ((2**(q+1)) * fact(q+1) * fact(alpha - 2*m)))
                        part_2 = (ell[0, d]**(4*m)) * (x[d, n]**(alpha - 2*m))
                        b += part_1 * part_2
                    temp[d] = a * b

                # final result
                result[n, q] = np.prod(temp)

        return result

    def der_par(self, par_0, x):  # K as kwarg would save computation (would have to be evaluated w/ par_0)
        # par_0: array_like [alpha, el_1, ..., el_D]
        # x: (D, N)
        par_0 = par_0.squeeze()
        alpha, el = par_0[0], par_0[1:]
        K = self.eval(par_0, x)
        # derivative w.r.t. alpha (N,N)
        d_alpha = 2 * alpha ** -1 * K
        # derivatives w.r.t. el_1, ..., el_D (N,N,D)
        d_el = (x[:, na, :] - x[:, :, na]) ** 2 * (el ** -2)[:, na, na] * K[na, :, :]
        return np.concatenate((d_alpha[..., na], d_el.T), axis=2)

    @staticmethod
    def _unpack_parameters(par):
        """
        Extract scaling parameter and square-root of inverse length-scale matrix from vector of kernel parameters.

        Parameters
        ----------
        par : (dim, ) ndarray

        Returns
        -------
        : tuple

        """
        par = par.astype(float).squeeze()
        # TODO: return scaling and length-scale, not sqrt inv lambda
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
        self.dof = dof
        self.x_samples = multivariate_t(mean, cov, dof, size=num_mc).T  # (D, MC)
        super(RBFStudent, self).__init__(dim, par, jitter)

    def exp_x_kx(self, par, x, scaling=False):
        return (1/self.num_mc) * self.eval(par, self.x_samples, x, scaling=scaling).sum(axis=0)

    def exp_x_xkx(self, par, x):
        k = self.eval(par, self.x_samples, x, scaling=False)  # (MC, N)
        return (1 / self.num_mc) * (self.x_samples[..., na] * k[na, ...]).sum(axis=1)

    def exp_x_kxkx(self, par_0, par_1, x, scaling=False):
        """
        Correlation matrix of kernels with elements

        .. math::
            \\mathbb{E}[k(x, x_i), k(x, x_j)] = \\int\\! k(x, x_i), k(x, x_j) N(x \\mid 0, I)\\, \\mathrm{d}x

        Parameters
        ----------
        x : (dim, N) ndarray
            Data points.

        par_0 : (dim, ) ndarray
        par_1 : (dim, ) ndarray
            Kernel parameters.

        scaling : bool
            Kernel scaling parameter used when `scaling=True`.

        Returns
        -------
        : ndarray
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


class RQ(Kernel):
    """
    Rational Quadratic kernel.

    .. math::
        k(x, x') = s^2 \\left( 1 + \\frac{1}{2\\alpha}(x - x')^{\\top}\\Lambda^{-1} (x - x') \\right)^{-\\alpha}

    Parameters
    ----------
    dim : int
        Input dimension.

    par : (dim_out, num_par) ndarray
        Kernel parameters in a matrix, where i-th row contains parameters for i-th output.
        Each row is :math:`[s, \\alpha, \\ell_1, \\ldots, \\ell_{dim}]`.

    jitter : float
        Jitter for stabilizing inversion of the kernel matrix. Default ``jitter=1e-8``.

    Notes
    -----
    The kernel expectations are w.r.t standard Student's t-density and are approximate.
    """

    def __init__(self, dim, par, jitter=1e-8):
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
        RQ kernel mean, where each element is given by

        .. math::
            \\mathbb{E}_{x}[k(x, x_i)] = \\int\\! k(x, x_i) St(x \\mid 0, I, \\nu) \\,\\mathrm{d}x

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        x : ndarray
            Data points.

        Returns
        -------
        : ndarray
            Kernel mean.
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
            \\mathbb{E}_{x}[xk(x, x_i)] = \\int\\! xk(x, x_i) St(x \\mid 0, I, \\nu)\\, \\mathrm{d}x

        Parameters
        ----------
        x : ndarray
            Sigma-points (data) in a 2d array of shape (dim, N).

        par : array_like
            Kernel parameters in a vector. The first element must be kernel scaling parameter.

        Returns
        -------
        : ndarray
            Expectation for given data points :math:`x_i` and vector of kernel parameters :math:`\\theta_m`
            returned in an array of shape `(D, N)`, where `(D, N) = x.shape`.
        """
        s, alpha, sqrt_inv_lam = RQ._unpack_parameters(par)
        lam = np.diag(sqrt_inv_lam.diagonal() ** -2)

        mu_q = la.inv(lam + self.eye_d).dot(x)
        q = self.exp_x_kx(par, x)
        return q[na, :] * mu_q

    def exp_x_kxkx(self, par_0, par_1, x, scaling=False):
        """
        RQ kernel correlation

        .. math::
            \\mathbb{E}[k(x, x_i), k(x, x_j)] = \\int\\! k(x, x_i), k(x, x_j) St(x \\mid 0, I, \\nu)\\, \\mathrm{d}x

        Parameters
        ----------
        x : (dim, N) ndarray
            Data points.

        par_0 : (dim, ) ndarray
        par_1 : (dim, ) ndarray
            Kernel parameters.

        scaling : bool
            Kernel scaling parameter used when `scaling=True`.

        Returns
        -------
        : ndarray
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
        """
        RQ kernel expectation

        .. math::
            \\mathbb{E}_x[k(x, x)] = \\int\\! k(x, x) St(x \\mid 0, I, \\nu)\\, \\mathrm{d}x

        Parameters
        ----------
        par : (dim, ) ndarray
            Kernel parameters.

        Returns
        -------
        : ndarray
            Kernel expectation.
        """
        return par[0] ** 2

    def exp_xy_kxy(self, par):
        """
        RQ kernel expectation :math:`\\mathbb{E}_{x, x'}[k(x, x')]`, where :math:`x,x' \\sim St(0, I, \\nu)`.

        Parameters
        ----------
        par : (dim, ) ndarray
            Kernel parameters.

        Returns
        -------
        : ndarray
            Correlation matrix of kernels computed for given pair of kernel parameters.
        """
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
        par : ndarray

        Returns
        -------
        : tuple

        """
        par = par.astype(float).squeeze()
        return par[0], par[1], np.diag(par[2:] ** -1)
