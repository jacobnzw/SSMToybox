from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from numpy.polynomial.hermite_e import hermegauss, hermeval
from scipy.linalg import cho_factor, cho_solve, block_diag
from scipy.io import loadmat, savemat
from scipy.special import factorial
from scipy.optimize import minimize
from sklearn.utils.extmath import cartesian
from ssmtoybox.utils import log_cred_ratio, mse_matrix, bigauss_mixture, multivariate_t, maha
from figprint import *
import joblib


# Inference
class StateSpaceInference(metaclass=ABCMeta):
    def __init__(self, ssm, tf_dyn, tf_meas):

        # state-space model of a dynamical system whose state is to be estimated
        assert isinstance(ssm, StateSpaceModel)
        self.ssm = ssm

        # separate moment transforms for system dynamics and measurement model
        assert isinstance(tf_dyn, MomentTransform) and isinstance(tf_meas, MomentTransform)
        self.tf_dyn = tf_dyn
        self.tf_meas = tf_meas

        self.flags = {'filtered': False, 'smoothed': False}
        self.x_mean_pr, self.x_cov_pr, = None, None
        self.x_mean_sm, self.x_cov_sm = None, None
        self.xx_cov, self.xy_cov = None, None
        self.pr_mean, self.pr_cov, self.pr_xx_cov = None, None, None
        self.fi_mean, self.fi_cov = None, None
        self.sm_mean, self.sm_cov = None, None
        self.D, self.N = None, None

    def get_flag(self, key):
        return self.flags[key]

    def set_flag(self, key, value):
        self.flags[key] = value

    def forward_pass(self, data):
        self.D, self.N = data.shape
        self.fi_mean = np.zeros((self.ssm.xD, self.N+1))
        self.fi_cov = np.zeros((self.ssm.xD, self.ssm.xD, self.N+1))
        # FIXME: saving initial conditions to filtered state is redundant
        # NOTE: if init. conds must be saved (smoother?) than fi_mean should be one larger than # measurements to
        # accommodate inits.

        # first step == initial conditions
        self.fi_mean[:, 0], self.fi_cov[..., 0] = self.x_mean_fi, self.x_cov_fi
        self.pr_mean = self.fi_mean.copy()
        self.pr_cov = self.fi_cov.copy()
        self.pr_xx_cov = self.fi_cov.copy()

        # pad data with zeros so that indices align with states
        data = np.hstack((np.zeros((self.D, 1)), data))

        for k in range(1, self.N+1):  # iterate over columns of data

            # compute predicted moments
            self._time_update(k - 1)
            self.pr_mean[..., k] = self.x_mean_pr
            self.pr_cov[..., k] = self.x_cov_pr
            self.pr_xx_cov[..., k] = self.xx_cov

            # compute filtered moments
            self._measurement_update(data[:, k], k)
            self.fi_mean[..., k], self.fi_cov[..., k] = self.x_mean_fi, self.x_cov_fi

        # set flag that filtered state sequence is available
        self.set_flag('filtered', True)

        # smoothing estimate at the last time step == the filtering estimate at the last time step
        self.x_mean_sm, self.x_cov_sm = self.x_mean_fi, self.x_cov_fi
        return self.fi_mean[:, 1:, ...], self.fi_cov[:, :, 1:, ...]

    def backward_pass(self):
        assert self.get_flag('filtered')  # require filtered state
        self.sm_mean = self.fi_mean.copy()
        self.sm_cov = self.fi_cov.copy()
        for k in range(self.N-2, 0, -1):
            self.x_mean_pr = self.pr_mean[..., k + 1]
            self.x_cov_pr = self.pr_cov[..., k + 1]
            self.xx_cov = self.pr_xx_cov[..., k+1]
            self.x_mean_fi = self.fi_mean[..., k]
            self.x_cov_fi = self.fi_cov[..., k]
            self._smoothing_update()
            self.sm_mean[..., k] = self.x_mean_sm
            self.sm_cov[..., k] = self.x_cov_sm
        self.set_flag('smoothed', True)
        return self.sm_mean, self.sm_cov

    def reset(self):
        self.x_mean_pr, self.x_cov_pr = None, None
        self.x_mean_sm, self.x_cov_sm = None, None
        self.xx_cov, self.xy_cov = None, None
        self.pr_mean, self.pr_cov, self.pr_xx_cov = None, None, None
        self.fi_mean, self.fi_cov = None, None
        self.sm_mean, self.sm_cov = None, None
        self.D, self.N = None, None
        self.flags = {'filtered': False, 'smoothed': False}

    @abstractmethod
    def _time_update(self, time, theta_dyn=None, theta_obs=None):
        pass

    @abstractmethod
    def _measurement_update(self, y, time=None):
        pass

    @abstractmethod
    def _smoothing_update(self):
        pass


class GaussianInference(StateSpaceInference):

    def __init__(self, ssm, tf_dyn, tf_meas):

        # dynamical system whose state is to be estimated
        assert isinstance(ssm, StateSpaceModel)

        # set initial condition mean and covariance, and noise covariances
        self.x_mean_fi, self.x_cov_fi, self.q_mean, self.q_cov, self.r_mean, self.r_cov, self.G = ssm.get_pars(
            'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov', 'q_gain'
        )

        super(GaussianInference, self).__init__(ssm, tf_dyn, tf_meas)

    def reset(self):
        self.x_mean_fi, self.x_cov_fi = self.ssm.get_pars('x0_mean', 'x0_cov')
        super(GaussianInference, self).reset()

    def _time_update(self, time, theta_dyn=None, theta_obs=None):
        # in non-additive case, augment mean and covariance
        mean = self.x_mean_fi if self.ssm.q_additive else np.hstack((self.x_mean_fi, self.q_mean))
        cov = self.x_cov_fi if self.ssm.q_additive else block_diag(self.x_cov_fi, self.q_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute predicted state mean, covariance
        self.x_mean_pr, self.x_cov_pr, self.xx_cov = self.tf_dyn.apply(self.ssm.dyn_eval, mean, cov,
                                                                       self.ssm.par_fcn(time), theta_dyn)
        if self.ssm.q_additive:
            self.x_cov_pr += self.G.dot(self.q_cov).dot(self.G.T)

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_pr if self.ssm.r_additive else np.hstack((self.x_mean_pr, self.r_mean))
        cov = self.x_cov_pr if self.ssm.r_additive else block_diag(self.x_cov_pr, self.r_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute measurement mean, covariance
        self.y_mean_pr, self.y_cov_pr, self.xy_cov = self.tf_meas.apply(self.ssm.meas_eval, mean, cov,
                                                                        self.ssm.par_fcn(time), theta_obs)
        # in additive case, noise covariances need to be added
        if self.ssm.r_additive:
            self.y_cov_pr += self.r_cov

        # in non-additive case, cross-covariances must be trimmed (has no effect in additive case)
        self.xy_cov = self.xy_cov[:, :self.ssm.xD]
        self.xx_cov = self.xx_cov[:, :self.ssm.xD]

    def _measurement_update(self, y, time=None):
        gain = cho_solve(cho_factor(self.y_cov_pr), self.xy_cov).T
        self.x_mean_fi = self.x_mean_pr + gain.dot(y - self.y_mean_pr)
        self.x_cov_fi = self.x_cov_pr - gain.dot(self.y_cov_pr).dot(gain.T)

    def _smoothing_update(self):
        gain = cho_solve(cho_factor(self.x_cov_pr), self.xx_cov).T
        self.x_mean_sm = self.x_mean_fi + gain.dot(self.x_mean_sm - self.x_mean_pr)
        self.x_cov_sm = self.x_cov_fi + gain.dot(self.x_cov_sm - self.x_cov_pr).dot(gain.T)


class StudentInference(StateSpaceInference):
    """
    Base class for state-space inference algorithms, which assume that the state and measurement variables are jointly
    Student distributed.

    Note, that even though Student's t distribution is not parametrized by the covariance matrix like the Gaussian,
    the filter still produces mean and covariance of the state.
    """

    def __init__(self, ssm, tf_dyn, tf_meas, dof=4.0, fixed_dof=True):
        """

        Parameters
        ----------
        ssm : StateSpaceModel
            State space model to perform inference on. Must implement the 'q_dof' and 'r_dof' properties.
        tf_dyn : MomentTransform
            Moment transform for system dynamics.
        tf_meas : MomentTransform
            Moment transform for measurement function.
        dof : float
            Degree of freedom parameter of the filtered density.
        fixed_dof : bool
            If `True`, DOF will be fixed for all time steps, which preserves the heavy-tailed behaviour of the filter.
            If `False`, DOF will be increasing after each measurement update, which means the heavy-tailed behaviour is
            not preserved and therefore converges to a Gaussian filter.
        """

        assert isinstance(ssm, StudentStateSpaceModel)

        # extract SSM parameters
        # initial statistics are taken to be filtered statistics
        self.x_mean_fi, self.x_cov_fi, self.dof_fi = ssm.get_pars('x0_mean', 'x0_cov', 'x0_dof')

        # state noise statistics
        self.q_mean, self.q_cov, self.q_dof, self.q_gain = ssm.get_pars('q_mean', 'q_cov', 'q_dof', 'q_gain')

        # measurement noise statistics
        self.r_mean, self.r_cov, self.r_dof = ssm.get_pars('r_mean', 'r_cov', 'r_dof')

        # scale matrix variables
        scale = (dof - 2)/dof
        self.x_smat_fi = scale * self.x_cov_fi  # turn initial covariance into initial scale matrix
        self.q_smat = scale * self.q_cov
        self.r_smat = scale * self.r_cov
        self.x_smat_pr, self.y_smat_pr, self.xy_smat = None, None, None

        self.dof = dof
        self.fixed_dof = fixed_dof

        super(StudentInference, self).__init__(ssm, tf_dyn, tf_meas)

    def reset(self):
        self.x_mean_fi, self.x_cov_fi, self.dof_fi = self.ssm.get_pars('x0_mean', 'x0_cov', 'x0_dof')
        scale = (self.dof - 2) / self.dof
        self.x_smat_fi = scale * self.x_cov_fi
        self.x_smat_pr, self.y_smat_pr, self.xy_smat = None, None, None
        super(StudentInference, self).reset()

    def _time_update(self, time, theta_dyn=None, theta_obs=None):

        if self.fixed_dof:  # fixed-DOF version

            # pick the smallest DOF
            dof_pr = np.min((self.dof_fi, self.q_dof, self.r_dof))

            # rescale filtered scale matrix?
            scale = (dof_pr - 2) / dof_pr
            # self.x_smat_fi = self.x_smat_fi * scale * self.dof_fi / (self.dof_fi - 2)

        else:  # increasing DOF version
            scale = (self.dof - 2) / self.dof

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_fi if self.ssm.q_additive else np.hstack((self.x_mean_fi, self.q_mean))
        smat = self.x_smat_fi if self.ssm.q_additive else block_diag(self.x_smat_fi, self.q_smat)
        assert mean.ndim == 1 and smat.ndim == 2

        # predicted state statistics
        self.x_mean_pr, self.x_cov_pr, self.xx_cov = self.tf_dyn.apply(self.ssm.dyn_eval, mean, smat,
                                                                       self.ssm.par_fcn(time), theta_dyn)
        # predicted covariance -> predicted scale matrix
        self.x_smat_pr = scale * self.x_cov_pr

        if self.ssm.q_additive:
            self.x_cov_pr += self.q_gain.dot(self.q_cov).dot(self.q_gain.T)
            self.x_smat_pr += self.q_gain.dot(self.q_smat).dot(self.q_gain.T)

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_pr if self.ssm.r_additive else np.hstack((self.x_mean_pr, self.r_mean))
        smat = self.x_smat_pr if self.ssm.r_additive else block_diag(self.x_smat_pr, self.r_smat)
        assert mean.ndim == 1 and smat.ndim == 2

        # predicted measurement statistics
        self.y_mean_pr, self.y_cov_pr, self.xy_cov = self.tf_meas.apply(self.ssm.meas_eval, mean, smat,
                                                                        self.ssm.par_fcn(time), theta_obs)
        # turn covariance to scale matrix
        self.y_smat_pr = scale * self.y_cov_pr
        self.xy_smat = scale * self.xy_cov

        # in additive case, noise covariances need to be added
        if self.ssm.r_additive:
            self.y_cov_pr += self.r_cov
            self.y_smat_pr += self.r_smat

        # in non-additive case, cross-covariances must be trimmed (has no effect in additive case)
        self.xy_cov = self.xy_cov[:, :self.ssm.xD]
        self.xx_cov = self.xx_cov[:, :self.ssm.xD]
        self.xy_smat = self.xy_smat[:, :self.ssm.xD]

    def _measurement_update(self, y, time=None):

        # scale the covariance matrices
        # scale = (self.dof - 2) / self.dof
        # self.y_cov_pr *= scale
        # self.xy_cov *= scale

        # Kalman update
        gain = cho_solve(cho_factor(self.y_smat_pr), self.xy_smat).T
        self.x_mean_fi = self.x_mean_pr + gain.dot(y - self.y_mean_pr)
        self.x_cov_fi = self.x_smat_pr - gain.dot(self.y_smat_pr).dot(gain.T)

        # filtered covariance to filtered scale matrix
        # delta = cho_solve(cho_factor(self.y_smat_pr), y - self.y_mean_pr)
        delta = la.solve(la.cholesky(self.y_smat_pr), y - self.y_mean_pr)
        scale = (self.dof + delta.T.dot(delta)) / (self.dof + self.ssm.zD)
        self.x_smat_fi = scale * self.x_cov_fi

        # update degrees of freedom
        self.dof_fi += self.ssm.zD

    def _smoothing_update(self):
        # Student smoother has not been developed yet.
        pass


# Kernels
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
        d_el = (x[:, na, :] - x[:, :, na]) ** 2 * (el ** -2)[:, na, na] * K[na, :, :]
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


# Models
class Model(object, metaclass=ABCMeta):
    """
    A parent class for all models of the integrated function in the BQ quadrature context. It is intended to be used
    by the subclasses of the `BQTransform` (i.e. Gaussian process and t-process quadrature moment transforms). The
    Model class ties together the kernel and the point-set used by the underlying quadrature rule. In modelling
    terms, the Model is composed of a kernel and point-set, that is, `Model` *has-a* `Kernel` and `points`.

    Assumptions
    -----------
      - The model of the integrand relies on a Kernel class, that is, it is either a GP or TP regression model.

    Attributes
    ----------
    Model._supported_points_ : list
        Each element of the list is an acronym of a point-set.
    Model._supported_kernels_ : list
        Each element of the list is an acronym of a kernel.
    kernel : Kernel
        Kernel used by the Model.
    points : numpy.ndarray
        Quadrature rule point-set.
    str_pts : string
    str_pts_par : string
        String representation of the kernel parameter values.
    emv : float
        Expected model variance.
    ivar : float
        Variance of the integral.
    dim_in : int
        Dimension of the point-set.
    num_pts : int
        Number of points.
    eye_d
    eye_n : numpy.ndarray
        Pre-allocated identity matrices to ease the computations.
    """

    _supported_points_ = ['sr', 'ut', 'gh', 'fs']
    _supported_kernels_ = ['rbf', 'rq', 'rbf-student']

    def __init__(self, dim, kern_par, kernel, points, point_par=None):
        """
        Initialize model of the integrand with specified kernel and point set.

        Parameters
        ----------
        dim : int
            Dimension of the points (integration domain).
        kern_par : numpy.ndarray
            Kernel parameters in a vector.
        kernel : string
            String abbreviation for the kernel.
        points : string
            String abbreviation for the point-set.
        point_par : dict
            Any parameters for constructing desired point-set.
        """

        # init kernel and sigma-points
        self.kernel = Model.get_kernel(dim, kernel, kern_par)
        self.points = Model.get_points(dim, points, point_par)

        # init variables for passing kernel expectations and kernel matrix inverse
        self.q, self.Q, self.R, self.iK = None, None, None, None

        # save for printing
        self.str_pts = points
        self.str_pts_par = str(point_par)

        # may no longer be necessary now that jitter is in kernel
        self.dim_in, self.num_pts = self.points.shape
        self.eye_d, self.eye_n = np.eye(self.dim_in), np.eye(self.num_pts)

    def __str__(self):
        """
        Prettier string representation.

        Returns
        -------
        string
            String representation including short name of the point-set, the kernel and its parameter values.
        """
        return '{}\n{} {}'.format(self.kernel, self.str_pts, self.str_pts_par)

    def bq_weights(self, par):
        """
        Weights of the Bayesian quadrature.

        Weights for both GPQ and TPQ are the same, hence they're implemented in the general model class.

        Parameters
        ----------
        par : array_like

        Returns
        -------
        : tuple
        Weights for computation of the transformed mean, covariance and cross-covariance in a tuple ``(wm, Wc, Wcc)``.

        """
        par = self.kernel.get_parameters(par)
        x = self.points

        # inverse kernel matrix
        iK = self.kernel.eval_inv_dot(par, x, scaling=False)

        # Kernel expectations
        q = self.kernel.exp_x_kx(par, x)
        Q = self.kernel.exp_x_kxkx(par, par, x)
        R = self.kernel.exp_x_xkx(par, x)

        # save for EMV and IVAR computation
        self.q, self.Q, self.R, self.iK = q, Q, R, iK

        # BQ weights in terms of kernel expectations
        w_m = q.dot(iK)
        w_c = iK.dot(Q).dot(iK)
        w_cc = R.dot(iK)

        # covariance weights should be symmetric
        w_c = 0.5 * (w_c + w_c.T)

        return w_m, w_c, w_cc

    @abstractmethod
    def predict(self, test_data, fcn_obs, par=None):
        """
        Model predictions based on test points and the kernel parameters.

        Notes
        -----
        This is an abstract method. Implementation needs to be provided by the subclass.

        Parameters
        ----------
        test_data : numpy.ndarray
            Test points where to generate data.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        par : numpy.ndarray
            Kernel parameters, default `par=None`.

        Returns
        -------
        (mean, var)
            Model predictive mean and variance at the test point locations.
        """
        pass

    @abstractmethod
    def exp_model_variance(self, fcn_obs):
        """
        Expected model variance given the function observations and the kernel parameters.

        Notes
        -----
        This is an abstract method. Implementation needs to be provided by the subclass and should be easily
        accomplished using the kernel expectation method from the `Kernel` class.

        Parameters
        ----------
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
        float
            Expected model variance.
        """
        pass

    @abstractmethod
    def integral_variance(self, fcn_obs, par=None):
        """
        Integral variance given the function value observations and the kernel parameters.

        Notes
        -----
        This is an abstract method. Implementation needs to be provided by the subclass and should be easily
        accomplished using the kernel expectation method from the `Kernel` class.

        Parameters
        ----------
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        par : numpy.ndarray
            Kernel parameters, default `par=None`.

        Returns
        -------
        float
            Variance of the integral.
        """
        pass

    @abstractmethod
    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative logarithm of marginal likelihood of the model given the kernel parameters and the function
        observations.

        Notes
        -----
        Intends to be used as an objective function passed into the optimizer, thus it needs to subscribe to certain
        implementation conventions.

        Parameters
        ----------
        log_par : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the inputs supplied in `x_obs`.
        x_obs : numpy.ndarray
            Function inputs.
        jitter : numpy.ndarray
            Regularization term for kernel matrix inversion.

        Returns
        -------
        float
            Negative log marginal likelihood.

        """
        pass

    def likelihood_reg_emv(self, log_par, fcn_obs):
        """
        Negative marginal log-likelihood with a expected model variance as regularizer.

        Parameters
        ----------
        log_par : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
            Sum of negative marginal log-likelihood and expected model variance.
        """
        # negative marginal log-likelihood w/ additional regularizing term
        # regularizing terms: integral variance, expected model variance or both, prior on par
        nlml, nlml_grad = self.neg_log_marginal_likelihood(log_par, fcn_obs)
        # NOTE: not entirely sure regularization is usefull, because the regularized ML-II seems to give very similar
        # results to ML-II; this regularizer tends to prefer longer lengthscales
        reg = self.exp_model_variance(fcn_obs)
        return nlml + reg

    def likelihood_reg_ivar(self, log_par, fcn_obs):
        """
        Negative marginal log-likelihood with a integral variance as regularizer.

        Parameters
        ----------
        log_par : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
            Sum of negative marginal log-likelihood and integral variance.
        """
        # negative marginal log-likelihood w/ additional regularizing term
        nlml, nlml_grad = self.neg_log_marginal_likelihood(log_par, fcn_obs)
        reg = self.integral_variance(fcn_obs, par=np.exp(log_par))
        return nlml + reg

    def optimize(self, log_par_0, fcn_obs, x_obs, crit='NLML', method='BFGS', **kwargs):
        """
        Find optimal values of kernel parameters by minimizing chosen criterion given the point-set and the function
        observations.

        Parameters
        ----------
        log_par_0 : numpy.ndarray
            Initial guess of the kernel log-parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        x_obs : numpy.ndarray
            Function inputs.
        crit : string
            Objective function to use as a criterion for finding optimal setting of kernel parameters. Possible
            values are:
              - 'nlml' : negative marginal log-likelihood,
              - 'nlml+emv' : NLML with expected model variance as regularizer,
              - 'nlml+ivar' : NLML with integral variance as regularizer.
        method : string
            Optimization method for `scipy.optimize.minimize`, default method='BFGS'.
        **kwargs
            Keyword arguments for the `scipy.optimize.minimize`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Results of the optimization in a dict-like structure returned by `scipy.optimize.minimize`.

        Notes
        -----
        The criteria using expected model variance and integral variance as regularizers ('nlml+emv', 'nlml+ivar')
        are somewhat experimental. I did not operate under any sound theoretical justification when implementing
        those. Just curious to see what happens, thus might be removed in the future.

        See Also
        --------
        scipy.optimize.minimize
        """
        crit = crit.lower()
        if crit == 'nlml':
            obj_func = self.neg_log_marginal_likelihood
            jac = True
        elif crit == 'nlml+emv':
            obj_func = self.likelihood_reg_emv
            jac = False  # gradients not implemented for regularizers (solver uses approximations)
        elif crit == 'nlml+ivar':
            obj_func = self.likelihood_reg_ivar
            jac = False  # gradients not implemented for regularizers (solver uses approximations)
        else:
            raise ValueError('Unknown criterion {}.'.format(crit))
        jitter = 1e-8 * np.eye(x_obs.shape[1])
        return minimize(obj_func, log_par_0, args=(fcn_obs, x_obs, jitter), method=method, jac=jac, **kwargs)

    def plot_model(self, test_data, fcn_obs, par=None, fcn_true=None, in_dim=0):
        """
        Plot of predictive mean and variance of the fitted model of the integrand. Since we're plotting a function with
        multiple inputs and outputs, we need to specify which is to be plotted.

        Notes
        -----
        Not tested very much, likely to misbehave.

        Parameters
        ----------
        test_data : numpy.ndarray
            1D array of locations, where the function is to be evaluated for plotting.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        par : numpy.ndarray
            Kernel parameters, default `par=None`.
        fcn_true :
            True function values
        in_dim : int
            Index of the input dimension to plot.

        Returns
        -------

        """
        assert in_dim <= self.dim_in - 1

        fcn_obs = np.squeeze(fcn_obs)
        fcn_true = np.squeeze(fcn_true)

        # model predictive mean and variance
        mean, var = self.predict(test_data, fcn_obs, par=par)
        std = np.sqrt(var)
        test_data = np.squeeze(test_data[in_dim, :])

        # set plot title according to model
        fig_title = self.__class__.__name__ + ' model of the integrand'

        # plot training data, predictive mean and variance
        fig = plt.figure(fig_title)
        plt.fill_between(test_data, mean - 2 * std, mean + 2 * std, color='0.1', alpha=0.15)
        plt.plot(test_data, mean, color='k', lw=2)
        plt.plot(self.points[in_dim, :], fcn_obs, 'ko', ms=8)

        # true function values at test points if provided
        if fcn_true is not None:
            plt.plot(test_data, fcn_true, lw=2, ls='--', color='tomato')
        plt.show()

    @staticmethod
    def get_points(dim, points, point_par):
        """
        Construct desired point-set for integration. Calls methods of classical quadrature classes.

        Parameters
        ----------
        dim : int

        points : string
            String abbreviation for the point-set.
        point_par : dict
            Parameters for constructing desired point-set.

        Returns
        -------
        numpy.ndarray
            Point set in (D, N) array, where D is dimension and N number of points.

        Notes
        -----
        List of supported points is kept in ``_supported_points_`` class variable.
        """

        points = points.lower()

        # make sure points is supported
        if points not in Model._supported_points_:
            print('Points {} not supported. Supported points are {}.'.format(points, Model._supported_points_))
            return None
        if point_par is None:
            point_par = {}

        # create chosen points
        if points == 'sr':
            return SphericalRadial.unit_sigma_points(dim)
        elif points == 'ut':
            return Unscented.unit_sigma_points(dim, **point_par)
        elif points == 'gh':
            return GaussHermite.unit_sigma_points(dim, **point_par)
        elif points == 'fs':
            return FullySymmetricStudent.unit_sigma_points(dim, **point_par)

    @staticmethod
    def get_kernel(dim, kernel, par):
        """
        Initializes desired kernel.

        Parameters
        ----------
        dim : int
            Dimension of input (integration domain).
        kernel : string
            String abbreviation of the kernel.
        par : numpy.ndarray
            Parameters of the kernel.

        Returns
        -------
        : Kernel
            A subclass of Kernel.

        Notes
        -----
        List of supported kernels is kept in ``_supported_kernels_`` class variable.
        """

        kernel = kernel.lower()

        # make sure kernel is supported
        if kernel not in Model._supported_kernels_:
            print('Kernel {} not supported. Supported kernels are {}.'.format(kernel, Model._supported_kernels_))
            return None

        # initialize the chosen kernel
        if kernel == 'rbf':
            return RBF(dim, par)
        elif kernel == 'rbf-student':
            return RBFStudent(dim, par)
        # elif kernel == 'rq':
        #     return RQ(dim, par)


class GaussianProcess(Model):  # consider renaming to GaussianProcessRegression/GPRegression, same for TP
    """
    Gaussian process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kern_par, kernel='rbf', points='ut', point_par=None):
        """
        Gaussian process regression model.

        Parameters
        ----------
        dim : int
            Number of input dimensions.
        kern_par : numpy.ndarray
            Kernel parameters in matrix.
        kernel : string
            Acronym of the covariance function of the Gaussian process model.
        points : string
            Acronym for the sigma-point set to use in BQ.
        point_par : dict
            Parameters of the sigma-point set.
        """

        super(GaussianProcess, self).__init__(dim, kern_par, kernel, points, point_par)

    def predict(self, test_data, fcn_obs, x_obs=None, par=None):
        """
        Gaussian process predictions.

        Parameters
        ----------
        test_data : numpy.ndarray
            Test data, shape (D, M)
        fcn_obs : numpy.ndarray
            Observations of the integrand at sigma-points.
        x_obs : numpy.ndarray
            Training inputs.
        par : numpy.ndarray
            Kernel parameters.

        Returns
        -------
        : tuple
            Predictive mean and variance in a tuple (mean, var).

        """

        if x_obs is None:
            x_obs = self.points

        par = self.kernel.get_parameters(par)

        iK = self.kernel.eval_inv_dot(par, x_obs)
        kx = self.kernel.eval(par, test_data, x_obs)
        kxx = self.kernel.eval(par, test_data, test_data, diag=True)

        # GP mean and predictive variance
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        return mean, var

    def exp_model_variance(self, fcn_obs):
        """

        Parameters
        ----------
        fcn_obs : numpy.ndarray
        Q : numpy.ndarray
        iK : numpy.ndarray

        Returns
        -------
        : float

        """

        return self.kernel.scale.squeeze() ** 2 * (1 - np.trace(self.Q.dot(self.iK)))

    def integral_variance(self, fcn_obs, par=None):
        """

        Parameters
        ----------
        fcn_obs : numpy.ndarray
        par : numpy.ndarray

        Returns
        -------
        : float

        """

        par = self.kernel.get_parameters(par)
        q = self.kernel.exp_x_kx(par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
        kbar = self.kernel.exp_xy_kxy(par)
        return kbar - q.T.dot(iK).dot(q)

    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative marginal log-likelihood of single-output Gaussian process regression model.

        The likelihood is given by

        .. math::
        \[
        -\log p(Y \mid X, \theta) = -\sum_{e=1}^{\mathrm{dim_out}} \log p(y_e \mid X, \theta)
        \]

        where :math:`y_e` is e-th column of :math:`Y`. We have the same parameters :math:`\theta` for all outputs,
        which is more limiting than the multi-output case. For single-output dimension the expression is equivalent to
        negative marginal log-likelihood.

        Parameters
        ----------
        log_par : numpy.ndarray
            Kernel log-parameters, shape (num_par, ).
        fcn_obs : numpy.ndarray
            Function values, shape (num_pts, dim_out).
        x_obs : numpy.ndarray
            Function inputs, shape ().
        jitter : numpy.ndarray
            Regularization term for kernel matrix inversion.

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.

        Returns
        -------
        Negative log-likelihood and gradient for given parameter.

        """

        # convert from log-par to par
        par = np.exp(log_par)
        num_data, num_out = fcn_obs.shape

        K = self.kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ji', fcn_obs.T, a)  # sum of diagonal of A.T.dot(A)
        a_out_a = np.einsum('i...j, ...jn', a, a.T)  # (N, N) sum over of outer products of columns of A

        # negative total NLML
        nlml = num_out * np.sum(np.log(np.diag(L[0]))) + 0.5 * (y_dot_a + num_out * num_data * np.log(2 * np.pi))

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_par(par, x_obs)  # (N, N, num_hyp)
        iKdK = la.cho_solve(L, dK_dTheta)

        # gradient of total NLML
        dnlml_dtheta = 0.5 * np.trace((num_out * iKdK - a_out_a.dot(dK_dTheta)))  # (num_par, )

        return nlml, dnlml_dtheta


class StudentTProcess(Model):
    """
    Student t process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kern_par, kernel='rbf', points='ut', point_par=None, nu=3.0):
        """
        Student t process regression model.

        Parameters
        ----------
        dim : int
            Number of input dimensions.
        kern_par : numpy.ndarray
            Kernel parameters in matrix.
        kernel : string
            Acronym of the covariance function of the Gaussian process model.
        points : string
            Acronym for the sigma-point set to use in BQ.
        point_par : dict
            Parameters of the sigma-point set.
        nu : float
            Degrees of freedom.
        """

        super(StudentTProcess, self).__init__(dim, kern_par, kernel, points, point_par)
        nu = 3.0 if nu < 2 else nu  # nu > 2
        self.nu = nu

    def predict(self, test_data, fcn_obs, x_obs=None, par=None, nu=None):
        """
        Student t process predictions.

        Parameters
        ----------
        test_data : numpy.ndarray
            Test data, shape (D, M)
        fcn_obs : numpy.ndarray
            Observations of the integrand at sigma-points.
        x_obs : numpy.ndarray
            Training inputs.
        par : numpy.ndarray
            Kernel parameters.
        nu : float
            Degrees of freedom.

        Returns
        -------
        : tuple
            Predictive mean and variance in a tuple (mean, var).

        """

        par = self.kernel.get_parameters(par)
        if nu is None:
            nu = self.nu
        if x_obs is None:
            x_obs = self.points

        iK = self.kernel.eval_inv_dot(par, x_obs)
        kx = self.kernel.eval(par, test_data, x_obs)
        kxx = self.kernel.eval(par, test_data, test_data, diag=True)
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        scale = (nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (nu - 2 + self.num_pts)
        return mean, scale * var

    def exp_model_variance(self, fcn_obs):
        """

        Parameters
        ----------
        fcn_obs
        par
        Q
        iK

        Returns
        -------

        """

        fcn_obs = np.squeeze(fcn_obs)
        quad_form = np.einsum('ij,jk,ik->i', fcn_obs, self.iK, fcn_obs)
        scale = (self.nu - 2 + quad_form) / (self.nu - 2 + self.num_pts)
        return scale * self.kernel.scale.squeeze() ** 2 * (1 - np.trace(self.Q.dot(self.iK)))

    def integral_variance(self, fcn_obs, par=None):
        """

        Parameters
        ----------
        fcn_obs
        par

        Returns
        -------

        """

        par = self.kernel.get_parameters(par)

        kbar = self.kernel.exp_xy_kxy(par)
        q = self.kernel.exp_x_kx(par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)

        fcn_obs = np.squeeze(fcn_obs)
        scale = (self.nu - 2 + fcn_obs.dot(iK).dot(fcn_obs.T)) / (self.nu - 2 + self.num_pts)
        return scale * (kbar - q.T.dot(iK).dot(q))

    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative marginal log-likelihood of Student t process regression model.

        Parameters
        ----------
        log_par : numpy.ndarray
            Kernel log-parameters, shape (num_par, ).
        fcn_obs : numpy.ndarray
            Function values, shape (num_pts, dim_out).
        x_obs : numpy.ndarray
            Function inputs, shape ().
        jitter : numpy.ndarray
            Regularization term for kernel matrix inversion.

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.

        Returns
        -------
        Negative log-likelihood and gradient for given parameter.

        """

        # convert from log-par to par
        par = np.exp(log_par)
        num_data, num_out = fcn_obs.shape
        nu = self.nu

        K = self.kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ij -> j', fcn_obs, a)  # sum of diagonal of A.T.dot(A)

        # negative marginal log-likelihood
        from scipy.special import gamma
        half_logdet_K = np.sum(np.log(np.diag(L[0])))
        const = (num_data/2) * np.log((nu-2)*np.pi) - np.log(gamma((nu+num_data)/2)) + np.log(gamma(nu/2))
        log_sum = 0.5*(self.nu + num_data) * np.log(1 + y_dot_a/(self.nu - 2)).sum()
        nlml = log_sum + num_out*(half_logdet_K + const)

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_par(par, x_obs)  # (N, N, num_par)

        # gradient
        iKdK = la.cho_solve(L, dK_dTheta)
        scale = (self.nu + num_data) / (self.nu + y_dot_a - 2)
        a_out_a = np.einsum('j, i...j, ...jn', scale, a, a.T)  # (N, N) weighted sum of outer products of columns of A
        dnlml_dtheta = 0.5 * np.trace((num_out * iKdK - a_out_a.dot(dK_dTheta)))  # (num_par, )

        return nlml, dnlml_dtheta


# Moment Transforms
class MomentTransform(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        pass


class Taylor1stOrder(MomentTransform):
    def __init__(self, dim):
        self.dim = dim

    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        mean_f = f(mean, fcn_pars)
        jacobian_f = f(mean, fcn_pars, dx=True)
        jacobian_f = jacobian_f.reshape(len(mean_f), self.dim)
        cov_fx = jacobian_f.dot(cov)
        cov_f = cov_fx.dot(jacobian_f.T)
        return mean_f, cov_f, cov_fx


class SigmaPointTransform(MomentTransform):
    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        mean = mean[:, na]
        # form sigma-points from unit sigma-points
        x = mean + la.cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(f, 0, x, fcn_pars)
        # output mean
        mean_f = fx.dot(self.wm)
        # output covariance
        dfx = fx - mean_f[:, na]
        cov_f = dfx.dot(self.Wc).dot(dfx.T)
        # input-output covariance
        cov_fx = dfx.dot(self.Wc).dot((x - mean).T)
        return mean_f, cov_f, cov_fx


class Unscented(SigmaPointTransform):
    """
    General purpose class implementing Unscented transform.
    """

    def __init__(self, dim, kappa=None, alpha=1.0, beta=2.0):
        # UT weights
        self.wm, self.wc = self.weights(dim, kappa=kappa, alpha=alpha, beta=beta)
        self.Wm = np.diag(self.wm)
        self.Wc = np.diag(self.wc)
        # UT unit sigma-points
        self.unit_sp = self.unit_sigma_points(dim, kappa=kappa, alpha=alpha)

    @staticmethod
    def unit_sigma_points(dim, kappa=None, alpha=1.0):
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa
        lam = alpha ** 2 * (dim + kappa) - dim
        c = np.sqrt(dim + lam)
        return np.hstack((np.zeros((dim, 1)), c * np.eye(dim), -c * np.eye(dim)))

    @staticmethod
    def weights(dim, kappa=None, alpha=1.0, beta=2.0):
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa
        lam = alpha ** 2 * (dim + kappa) - dim
        wm = 1.0 / (2.0 * (dim + lam)) * np.ones(2 * dim + 1)
        wc = wm.copy()
        wm[0] = lam / (dim + lam)
        wc[0] = wm[0] + (1 - alpha ** 2 + beta)
        return wm, wc


class SphericalRadial(SigmaPointTransform):
    # Could be implemented with Unscented with kappa=0, alpha=1, beta=0.
    def __init__(self, dim):
        self.wm = self.weights(dim)
        self.Wc = np.diag(self.wm)
        self.unit_sp = self.unit_sigma_points(dim)

    @staticmethod
    def weights(dim):
        return (1 / (2.0 * dim)) * np.ones(2 * dim)

    @staticmethod
    def unit_sigma_points(dim):
        c = np.sqrt(dim)
        return np.hstack((c * np.eye(dim), -c * np.eye(dim)))


class GaussHermite(SigmaPointTransform):
    def __init__(self, dim, degree=3):
        self.degree = degree
        self.wm = self.weights(dim, degree)
        self.Wc = np.diag(self.wm)
        self.unit_sp = self.unit_sigma_points(dim, degree)

    @staticmethod
    def weights(dim, degree=3):
        # 1D sigma-points (x) and weights (w)
        x, w = hermegauss(degree)
        # hermegauss() provides weights that cause posdef errors
        w = factorial(degree) / (degree ** 2 * hermeval(x, [0] * (degree - 1) + [1]) ** 2)
        return np.prod(cartesian([w] * dim), axis=1)

    @staticmethod
    def unit_sigma_points(dim, degree=3):
        # 1D sigma-points (x) and weights (w)
        x, w = hermegauss(degree)
        # nD sigma-points by cartesian product
        return cartesian([x] * dim).T  # column/sigma-point


class FullySymmetricStudent(SigmaPointTransform):
    """
    Moment transform for Student-t distributions based on fully symmetric integration rule from [1]_. The weights are
    coded for rule orders (degrees) 3 and 5 only. The 3rd order weights converge to UT weights for nu -> \infty.

    .. [1] J. McNamee and F. Stenger, “Construction of fully symmetric numerical integration formulas,”
           Numer. Math., vol. 10, no. 4, pp. 327–344, 1967.
    """

    _supported_degrees_ = [3, 5]

    def __init__(self, dim, degree=3, kappa=None, dof=4):
        """
        Initialize moment transform for Student distributed random variables based on fully-symmetric quadrature rule.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable (Dimension of the integration domain).
        degree : int
            Degree (order) of the quadrature rule.
        kappa : float
            Tuning parameter of the fully-symmetric point set. If `kappa=None`, chooses `kappa = max(3-dim, 0)`.
        dof : float
            Degree of freedom of the input density.
        """

        # init parameters stored in object variables
        self.degree, self.kappa, self.dof = degree, kappa, dof

        # init weights
        self.wm = self.weights(dim, degree, kappa, dof)
        self.Wc = np.diag(self.wm)

        # init unit sigma-points
        self.unit_sp = self.unit_sigma_points(dim, degree, kappa, dof)

    @staticmethod
    def weights(dim, degree=3, kappa=None, dof=4.0):
        """
        Weights of the fully symmetric rule for Student-t distribution.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable (Dimension of the integration domain)
        degree : int
            Order of the quadrature rule, only `degree=3` or `degree=5` implemented.
        kappa : float
            Tuning parameter controlling spread of points from the center.
        dof : float
            Degree of freedom parameter for the Student distribution.

        Returns
        -------

        """

        if degree not in FullySymmetricStudent._supported_degrees_:
            print("Defaulting to degree 3. Supplied degree {} not supported. Supported degrees: {}", degree,
                  FullySymmetricStudent._supported_degrees_)
            degree = 3

        # use kappa = 3 - dim if kappa not given
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa

        # dof > 2p, where degree = 2p+1
        dof = np.max((dof, degree))

        if degree == 3:  # code for 3rd-order rule

            # number of points for 3rd-order rule
            n = 2*dim + 1

            # weights are parametrized so that ST-3 -> UT-3 for dof -> inf
            w = 1 / (2 * (dim + kappa)) * np.ones(n)
            w[0] = kappa / (dim + kappa)
            return w

        else:  # code for 5th-order rule

            # compute weights in accordance to McNamee & Stenger (1967)
            I0 = 1
            I2 = dof / (dof - 2)
            I22 = dof ** 2 / ((dof - 2) * (dof - 4))
            I4 = 3 * I22

            A0 = I0 - dim * (I2 / I4) ** 2 * (I4 - 0.5 * (dim - 1) * I22)
            A1 = 0.5 * (I2 / I4) ** 2 * (I4 - (dim - 1) * I22)
            A11 = 0.25 * (I2 / I4) ** 2 * I22

            return np.hstack((A0, A1 * np.ones(2*dim), A11 * np.ones(2*dim*(dim-1))))

    @staticmethod
    def unit_sigma_points(dim, degree=3, kappa=None, dof=4.0):
        """
        Fully-symmetric unit sigma-point set.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable (dimension of the integration domain).
        degree : int
            Order of the quadrature rule, only `degree=3` or `degree=5` implemented.
        kappa : float
            Tuning parameter controlling spread of points from the center.
            If `kappa=None`, chooses `kappa = max(3-dim, 0)`.
        dof : float
            Degree of freedom parameter of the input density.

        Returns
        -------
        : numpy.ndarray
            Shape (dim, num_pts)

        """

        if degree not in FullySymmetricStudent._supported_degrees_:
            print("Defaulting to degree 3. Supplied degree {} not supported. Supported degrees: {}", degree,
                  FullySymmetricStudent._supported_degrees_)
            degree = 3

        # use kappa = 3 - dim if kappa not given
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa

        # dof > 2p, where degree = 2p+1
        dof = np.max((dof, degree))

        if degree == 3:  # code for 3rd order rule

            # pre-computed integrals, check McNamee & Stenger, 1967
            I2 = dof / (dof - 2)
            u = np.sqrt(I2 * (dim + kappa))
            return u * np.hstack((np.zeros((dim, 1)), np.eye(dim), -np.eye(dim)))

        else:  # code for 5th-order rule

            I2 = dof / (dof - 2)
            I4 = 3 * dof ** 2 / ((dof - 2) * (dof - 4))
            u = np.sqrt(I4 / I2)

            sp0 = FullySymmetricStudent.symmetric_set(dim, [])
            sp1 = FullySymmetricStudent.symmetric_set(dim, [u])
            sp2 = FullySymmetricStudent.symmetric_set(dim, [u, u])

            return np.hstack((sp0, sp1, sp2))

    @staticmethod
    def symmetric_set(dim, gen):
        """
        Symmetric point set.

        Parameters
        ----------
        dim : int
            Dimension
        gen : array_like (1 dimensional)
            Generator

        Notes
        -----
        Unscented transform points can be recovered by
            a0 = symmetric_set(dim, [])
            a1 = symmetric_set(dim, [1])
            ut = np.hstack((a0, a1))

        Returns
        -------

        """

        # if generator has no element
        nzeros = np.zeros((dim, 1))
        if not gen:
            return nzeros

        gen = np.asarray(gen)
        assert gen.ndim == 1, "Generator must be in 1d array_like."

        uind = np.arange(dim)  # indices of variable u for easier indexing
        eps = np.spacing(1.0)  # machine precision for comparisons
        sp = np.empty(shape=(dim, 0))

        for i in range(dim):
            u = nzeros.copy()
            u[i] = gen[0]

            if len(gen) > 1:
                if np.abs(gen[0] - gen[1]) < eps:
                    V = FullySymmetricStudent.symmetric_set(dim-i-1, gen[1:])
                    for j in range(V.shape[1]):
                        u[i+1:, 0] = V[:, j]
                        sp = np.hstack((sp, u, -u))
                else:
                    V = FullySymmetricStudent.symmetric_set(dim-1, gen[1:])
                    for j in range(V.shape[1]):
                        u[uind != i, 0] = V[:, j]
                        sp = np.hstack((sp, u, -u))
            else:
                sp = np.hstack((sp, u, -u))

        return sp


# BQ Moment Transforms
class BQTransform(MomentTransform, metaclass=ABCMeta):

    # list of supported models for the integrand
    _supported_models_ = ['gp', 'gp-mo', 'tp', 'tp-mo']  # mgp, gpder, ...

    def __init__(self, dim_in, dim_out, kern_par, model, kernel, points, point_par, **kwargs):
        self.model = BQTransform._get_model(dim_in, dim_out, model, kernel, points, kern_par, point_par, **kwargs)

        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self._weights()
        self.I_out = np.eye(dim_out)

    def apply(self, f, mean, cov, fcn_par, kern_par=None):
        """
        Compute transformed moments.

        Parameters
        ----------
        f : func
            Integrand, transforming function of the random variable.
        mean : numpy.ndarray
            Input mean.
        cov : numpy.ndarray
            Input covariance.
        fcn_par : numpy.ndarray
            Integrand parameters.
        kern_par : numpy.ndarray
            Kernel parameters.

        Returns
        -------
        : tuple
            Transformed mean, covariance and cross-covariance in tuple.

        """

        # re-compute weights if transform parameter kern_par explicitly given
        if kern_par is not None:
            self.wm, self.Wc, self.Wcc = self._weights(kern_par)

        mean = mean[:, na]
        chol_cov = la.cholesky(cov)

        # evaluate integrand at sigma-points
        x = mean + chol_cov.dot(self.model.points)
        fx = self._fcn_eval(f, x, fcn_par)

        # DEBUG
        self.fx = fx

        # compute transformed moments
        mean_f = self._mean(self.wm, fx)
        cov_f = self._covariance(self.Wc, fx, mean_f)
        cov_fx = self._cross_covariance(self.Wcc, fx, chol_cov)

        return mean_f, cov_f, cov_fx

    @staticmethod
    def _get_model(dim_in, dim_out, model, kernel, points, kern_par, point_par, **kwargs):
        """
        Initialize chosen model with supplied parameters.

        Parameters
        ----------
        dim_in : int
        dim_out : int
        model : string
            Model of the integrand. See `BQTransform._supported_models_`.
        kernel : string
            Kernel of the model. See `Model._supported_kernels_`.
        points : string
            Point-set to use for the integration. See `Model._supported_points_`.
        kern_par : numpy.ndarray
            Kernel parameters.
        point_par : dict
            Parameters of the point-set scheme.
        kwargs : dict
            Additional kwargs passed to the model.

        Returns
        -------
        : Model

        """

        # import must be after SigmaPointTransform
        # from .bqmodel import GaussianProcess, StudentTProcess, GaussianProcessMO, StudentTProcessMO
        model = model.lower()

        # make sure kernel is supported
        if model not in BQTransform._supported_models_:
            print('Model {} not supported. Supported models are {}.'.format(model, BQTransform._supported_models_))
            return None

        # initialize the chosen model
        if model == 'gp':
            return GaussianProcess(dim_in, kern_par, kernel, points, point_par)
        elif model == 'tp':
            return StudentTProcess(dim_in, kern_par, kernel, points, point_par, **kwargs)
        # elif model == 'gp-mo':
        #     return GaussianProcessMO(dim_in, dim_out, kern_par, kernel, points, point_par)
        # elif model == 'tp-mo':
        #     return StudentTProcessMO(dim_in, dim_out, kern_par, kernel, points, point_par, **kwargs)

    def minimum_variance_points(self, x0, kern_par):
        # run optimizer to find minvar point sets using initial guess x0; requires implemented _integral_variance()
        pass

    def _weights(self, kern_par=None):
        """
        Bayesian quadrature weights.

        Parameters
        ----------
        kern_par : numpy.ndarray
            Kernel parameters to use in computation of the weights.

        Returns
        -------
        : tuple
            Weights for the mean, covariance and cross-covariance quadrature approximations.

        """
        return self.model.bq_weights(kern_par)

    @abstractmethod
    def _integral_variance(self, points, kern_par):
        # can serve for finding minimum variance point sets or kernel parameters
        # optimizers require the first argument to be the variable, a decorator could be used to interchange the first
        # two arguments, so that we don't have to define the same function twice only w/ different signature
        pass

    @abstractmethod
    def _fcn_eval(self, fcn, x, fcn_par):
        """
        Evaluations of the integrand, which can comprise function observations as well as derivative observations.

        Parameters
        ----------
        fcn : func
            Integrand as a function handle, which is expected to behave certain way.
        x : numpy.ndarray
            Argument (input) of the integrand.
        fcn_par :
            Parameters of the integrand.
        Notes
        -----
        Methods in derived subclasses decides whether to return derivatives also

        Returns
        -------
        : numpy.ndarray
            Function evaluations of shape (out_dim, num_pts).

        """
        pass

    def _mean(self, weights, fcn_evals):
        """
        Transformed mean.

        Parameters
        ----------
        weights : numpy.ndarray
        fcn_evals : numpy.ndarray

        Returns
        -------
        : numpy.ndarray

        """
        return fcn_evals.dot(weights)
        # return np.einsum('en, n -> e', fcn_evals, weights)

    def _covariance(self, weights, fcn_evals, mean_out):
        """
        Transformed covariance.

        Parameters
        ----------
        weights : numpy.ndarray
        fcn_evals : numpy.ndarray
        mean_out : numpy.ndarray

        Returns
        -------
        : numpy.ndarray

        """
        expected_model_var = self.model.exp_model_variance(fcn_evals)
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var
        # return np.einsum('in, nm, jm -> ij', fcn_evals, weights, fcn_evals) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _cross_covariance(self, weights, fcn_evals, chol_cov_in):
        """
        Cross-covariance of input variable and transformed output variable.

        Parameters
        ----------
        weights : numpy.ndarray
            Shape (D, N)
        fcn_evals : numpy.ndarray
            Shape (E, N)
        chol_cov_in : numpy.ndarray
            Shape (D, D)

        Returns
        -------
        : numpy.ndarray

        """
        return fcn_evals.dot(weights.T).dot(chol_cov_in.T)
        # return np.einsum('en, dn, dj -> ej', fcn_evals, weights, chol_cov_in)

    def __str__(self):
        return '{}\n{}'.format(self.__class__.__name__, self.model)


class GPQ(BQTransform):  # consider renaming to GPQTransform
    def __init__(self, dim_in, dim_out, kern_par, kernel='rbf', points='ut', point_par=None):
        super(GPQ, self).__init__(dim_in, dim_out, kern_par, 'gp', kernel, points, point_par)

    def _fcn_eval(self, fcn, x, fcn_par):
        return np.apply_along_axis(fcn, 0, x, fcn_par)

    def _covariance(self, weights, fcn_evals, mean_out):
        """
        Transformed covariance.

        Parameters
        ----------
        weights : numpy.ndarray
        fcn_evals : numpy.ndarray
        mean_out : numpy.ndarray

        Returns
        -------
        : numpy.ndarray

        """
        expected_model_var = self.model.exp_model_variance(fcn_evals) * self.I_out
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _integral_variance(self, points, kern_par):
        pass


class TPQ(BQTransform):
    def __init__(self, dim_in, dim_out, kern_par, kernel='rbf', points='ut', point_par=None, nu=3.0):
        super(TPQ, self).__init__(dim_in, dim_out, kern_par, 'tp', kernel, points, point_par, nu=nu)

    def _fcn_eval(self, fcn, x, fcn_par):
        return np.apply_along_axis(fcn, 0, x, fcn_par)

    def _covariance(self, weights, fcn_evals, mean_out):
        expected_model_var = np.diag(self.model.exp_model_variance(fcn_evals))
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _integral_variance(self, points, kern_par):
        pass


# Gaussian Filters
class UnscentedKalman(GaussianInference):
    """
    Unscented Kalman filter and smoother.
    """

    def __init__(self, sys, kappa=None, alpha=1.0, beta=2.0):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Unscented(nq, kappa=kappa, alpha=alpha, beta=beta)
        th = Unscented(nr, kappa=kappa, alpha=alpha, beta=beta)
        super(UnscentedKalman, self).__init__(sys, tf, th)


# Studentian Filters
class ExtendedStudent(StudentInference):

    def __init__(self, sys, dof=4.0, fixed_dof=True):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Taylor1stOrder(nq)
        th = Taylor1stOrder(nr)
        super(ExtendedStudent, self).__init__(sys, tf, th, dof, fixed_dof)


class FSQStudent(StudentInference):
    """Filter based on fully symmetric quadrature rules."""

    def __init__(self, ssm, degree=3, kappa=None, dof=4.0, fixed_dof=True):
        assert isinstance(ssm, StudentStateSpaceModel)

        # correct input dimension if noise non-additive
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD

        # degrees of freedom for SSM noises
        q_dof, r_dof = ssm.get_pars('q_dof', 'r_dof')

        # init moment transforms
        t_dyn = FullySymmetricStudent(nq, degree, kappa, q_dof)
        t_obs = FullySymmetricStudent(nr, degree, kappa, r_dof)
        super(FSQStudent, self).__init__(ssm, t_dyn, t_obs, dof, fixed_dof)


class GPQStudent(StudentInference):

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, point_hyp=None, dof=4.0, fixed_dof=True):
        """
        Student filter with Gaussian Process quadrature moment transforms using fully-symmetric sigma-point set.

        Parameters
        ----------
        ssm : StudentStateSpaceModel
        kern_par_dyn : numpy.ndarray
            Kernel parameters for the GPQ moment transform of the dynamics.
        kern_par_obs : numpy.ndarray
            Kernel parameters for the GPQ moment transform of the measurement function.
        point_hyp : dict
            Point set parameters with keys:
              * `'degree'`: Degree (order) of the quadrature rule.
              * `'kappa'`: Tuning parameter of controlling spread of sigma-points around the center.
        dof : float
            Desired degree of freedom for the filtered density.
        fixed_dof : bool
            If `True`, DOF will be fixed for all time steps, which preserves the heavy-tailed behaviour of the filter.
            If `False`, DOF will be increasing after each measurement update, which means the heavy-tailed behaviour is
            not preserved and therefore converges to a Gaussian filter.
        """
        assert isinstance(ssm, StudentStateSpaceModel)

        # correct input dimension if noise non-additive
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD

        # degrees of freedom for SSM noises
        q_dof, r_dof = ssm.get_pars('q_dof', 'r_dof')

        # add DOF of the noises to the sigma-point parameters
        if point_hyp is None:
                point_hyp = dict()
        point_hyp_dyn = point_hyp.copy()
        point_hyp_obs = point_hyp.copy()
        point_hyp_dyn.update({'dof': q_dof})
        point_hyp_obs.update({'dof': r_dof})

        # init moment transforms
        t_dyn = GPQ(nq, ssm.xD, kern_par_dyn, 'rbf-student', 'fs', point_hyp_dyn)
        t_obs = GPQ(nr, ssm.zD, kern_par_obs, 'rbf-student', 'fs', point_hyp_obs)
        super(GPQStudent, self).__init__(ssm, t_dyn, t_obs, dof, fixed_dof)


class TPQStudent(StudentInference):
    """
    T-process quadrature filter and smoother for the Student's t inference. Uses RQ kernel and fully-symmetric
    point-sets by default. RQ kernel expectations w.r.t. Student's t-density are expressed as a simplified scale
    mixture representation which facilitates analytical tractability.
    """

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, point_par=None, dof=4.0, fixed_dof=True, dof_tp=4.0):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD

        # degrees of freedom for SSM noises
        q_dof, r_dof = ssm.get_pars('q_dof', 'r_dof')

        # add DOF of the noises to the sigma-point parameters
        if point_par is None:
            point_par = dict()
        point_par_dyn = point_par.copy()
        point_par_obs = point_par.copy()
        point_par_dyn.update({'dof': q_dof})
        point_par_obs.update({'dof': r_dof})
        # TODO: finish fixing DOFs, DOF for TPQ and DOF for the filtered state.

        t_dyn = TPQ(nq, ssm.xD, kern_par_dyn, 'rbf-student', 'fs', point_par_dyn, nu=dof_tp)
        t_obs = TPQ(nr, ssm.zD, kern_par_obs, 'rbf-student', 'fs', point_par_obs, nu=dof_tp)
        super(TPQStudent, self).__init__(ssm, t_dyn, t_obs, dof, fixed_dof)


def rbf_student_mc_weights(x, kern, num_samples, num_batch):
    # MC approximated BQ weights using RBF kernel and Student density
    # MC computed by batches, because without batches we would run out of memory for large sample sizes

    assert isinstance(kern, RBFStudent)
    # kernel parameters and input dimensionality
    par = kern.par
    dim, num_pts = x.shape

    # inverse kernel matrix
    iK = kern.eval_inv_dot(kern.par, x, scaling=False)
    mean, scale, dof = np.zeros((dim, )), np.eye(dim), kern.dof

    # compute MC estimates by batches
    num_samples_batch = num_samples // num_batch
    q_batch = np.zeros((num_pts, num_batch, ))
    Q_batch = np.zeros((num_pts, num_pts, num_batch))
    R_batch = np.zeros((dim, num_pts, num_batch))
    for ib in range(num_batch):

        # multivariate t samples
        x_samples = multivariate_t(mean, scale, dof, num_samples_batch).T

        # evaluate kernel
        k_samples = kern.eval(par, x_samples, x, scaling=False)
        kk_samples = k_samples[:, na, :] * k_samples[..., na]
        xk_samples = x_samples[..., na] * k_samples[na, ...]

        # intermediate sums
        q_batch[..., ib] = k_samples.sum(axis=0)
        Q_batch[..., ib] = kk_samples.sum(axis=0)
        R_batch[..., ib] = xk_samples.sum(axis=1)

    # MC approximations == sum the sums divide by num_samples
    c = 1/num_samples
    q = c * q_batch.sum(axis=-1)
    Q = c * Q_batch.sum(axis=-1)
    R = c * R_batch.sum(axis=-1)

    # BQ moment transform weights
    wm = q.dot(iK)
    wc = iK.dot(Q).dot(iK)
    wcc = R.dot(iK)
    return wm, wc, wcc, Q


def eval_perf_scores(x, mf, Pf):
    xD, steps, mc_sims, num_filt = mf.shape

    # average RMSE over simulations
    rmse = np.sqrt(((x[..., na] - mf) ** 2).sum(axis=0))
    rmse_avg = rmse.mean(axis=1)

    reg = 1e-6 * np.eye(xD)

    # average inclination indicator over simulations
    lcr = np.empty((steps, mc_sims, num_filt))
    for f in range(num_filt):
        for k in range(steps):
            mse = mse_matrix(x[:, k, :], mf[:, k, :, f]) + reg
            for imc in range(mc_sims):
                lcr[k, imc, f] = log_cred_ratio(x[:, k, imc], mf[:, k, imc, f], Pf[..., k, imc, f], mse)
    lcr_avg = lcr.mean(axis=1)

    return rmse_avg, lcr_avg


def run_filters(filters, z):
    num_filt = len(filters)
    zD, steps, mc_sims = z.shape
    xD = filters[0].ssm.xD

    # init space for filtered mean and covariance
    mf = np.zeros((xD, steps, mc_sims, num_filt))
    Pf = np.zeros((xD, xD, steps, mc_sims, num_filt))

    # run filters
    for i, f in enumerate(filters):
        print('Running {} ...'.format(f.__class__.__name__))
        for imc in range(mc_sims):
            mf[..., imc, i], Pf[..., imc, i] = f.forward_pass(z[..., imc])
            f.reset()

    # return filtered mean and covariance
    return mf, Pf


# State-Space Models
class StateSpaceModel(metaclass=ABCMeta):

    xD = None  # state dimension
    zD = None  # measurement dimension
    qD = None  # state noise dimension
    rD = None  # measurement noise dimension

    q_additive = None  # True = state noise is additive, False = non-additive
    r_additive = None

    def __init__(self, **kwargs):
        self.pars = kwargs
        self.zero_q = np.zeros(self.qD)
        self.zero_r = np.zeros(self.rD)

    @abstractmethod
    def dyn_fcn(self, x, q, pars):
        """ System dynamics.

        Abstract method for the system dynamics.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like
            Parameters of the system dynamics

        Returns
        -------
        1-D numpy.ndarray of shape (self.xD,)
            system state in the next time step
        """
        pass

    @abstractmethod
    def meas_fcn(self, x, r, pars):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            system state
        r : 1-D array_like of shape (self.rD,)
            measurement noise
        pars : 1-D array_like
            parameters of the measurement model

        Returns
        -------
        1-D numpy.ndarray of shape (self.zD,)
            measurement of the state
        """
        pass

    @abstractmethod
    def par_fcn(self, time):
        """Parameter function of the system dynamics and measurement model.

        Abstract method for the parameter function of the whole state-space model. The implementation should ensure
        that the system dynamics parameters come before the measurement model parameters in the returned vector of
        parameters.

        Parameters
        ----------
        time : int
            Discrete time step

        Returns
        -------
        1-D numpy.ndarray of shape (self.pD,)
            Vector of parameters at a given time.
        """
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, q, pars):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the system dynamics, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.qD)
        """
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, pars):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        r : 1-D array_like of shape (self.qD,)
            Measurement noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.rD)
        """
        pass

    @abstractmethod
    def state_noise_sample(self, size=None):
        """
        Sample from a state noise distribution.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

        """
        pass

    @abstractmethod
    def measurement_noise_sample(self, size=None):
        """
        Sample from a measurement noise distribution.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

        """
        pass

    @abstractmethod
    def initial_condition_sample(self, size=None):
        """
        Sample from a distribution over the system initial conditions.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

        """
        pass

    def dyn_eval(self, xq, pars, dx=False):
        """Evaluation of the system dynamics according to noise additivity.

        Parameters
        ----------
        xq : 1-D array_like
            Augmented system state
        pars : 1-D array_like
            System dynamics parameters
        dx : bool
            * ``True``: Evaluates derivatives (Jacobian) of the system dynamics
            * ``False``: Evaluates system dynamics
        Returns
        -------
            Evaluated system dynamics or evaluated Jacobian of the system dynamics.
        """

        if self.q_additive:
            assert len(xq) == self.xD
            if dx:
                out = (self.dyn_fcn_dx(xq, self.zero_q, pars).T.flatten())
            else:
                out = self.dyn_fcn(xq, self.zero_q, pars)
        else:
            assert len(xq) == self.xD + self.qD
            x, q = xq[:self.xD], xq[-self.qD:]
            if dx:
                out = (self.dyn_fcn_dx(x, q, pars).T.flatten())
            else:
                out = self.dyn_fcn(x, q, pars)
        return out

    def meas_eval(self, xr, pars, dx=False):
        """Evaluation of the system dynamics according to noise additivity.

        Parameters
        ----------
        xr : 1-D array_like
            Augmented system state
        pars : 1-D array_like
            Measurement model parameters
        dx : bool
            * ``True``: Evaluates derivatives (Jacobian) of the measurement model
            * ``False``: Evaluates measurement model
        Returns
        -------
            Evaluated measurement model or evaluated Jacobian of the measurement model.
        """

        if self.r_additive:
            # assert len(xr) == self.xD
            if dx:
                out = (self.meas_fcn_dx(xr, self.zero_r, pars).T.flatten())
            else:
                out = self.meas_fcn(xr, self.zero_r, pars)
        else:
            assert len(xr) == self.xD + self.rD
            x, r = xr[:self.xD], xr[-self.rD:]
            if dx:
                out = (self.meas_fcn_dx(x, r, pars).T.flatten())
            else:
                out = self.meas_fcn(x, r, pars)
        return out

    def check_jacobians(self, h=1e-8):
        """Checks implemented Jacobians.

        Checks that both Jacobians are correctly implemented using numerical approximations.

        Parameters
        ----------
        h : float
            step size in derivative approximations

        Returns
        -------
            Prints the errors and user decides whether they're acceptable.
        """

        nq = self.xD if self.q_additive else self.xD + self.qD
        nr = self.xD if self.r_additive else self.xD + self.rD
        xq, xr = np.random.rand(nq), np.random.rand(nr)
        hq_diag, hr_diag = np.diag(h * np.ones(nq)), np.diag(h * np.ones(nr))
        assert hq_diag.shape == (nq, nq) and hr_diag.shape == (nr, nr)
        xqph, xqmh = xq[:, na] + hq_diag, xq[:, na] - hq_diag
        xrph, xrmh = xr[:, na] + hr_diag, xr[:, na] - hr_diag

        # allocate space for Jacobians
        fph = np.zeros((self.xD, nq))
        hph = np.zeros((self.zD, nr))
        fmh, hmh = fph.copy(), hph.copy()

        # approximate Jacobians by central differences
        par = self.par_fcn(1.0)
        for i in range(nq):
            fph[:, i] = self.dyn_eval(xqph[:, i], par)
            fmh[:, i] = self.dyn_eval(xqmh[:, i], par)
        for i in range(nr):
            hph[:, i] = self.meas_eval(xrph[:, i], par)
            hmh[:, i] = self.meas_eval(xrmh[:, i], par)
        jac_fx = (2 * h) ** -1 * (fph - fmh)
        jac_hx = (2 * h) ** -1 * (hph - hmh)

        # report approximation error
        print("Errors in Jacobians\n{}\n{}".format(np.abs(jac_fx - self.dyn_eval(xq, par, dx=True)),
                                                   np.abs(jac_hx - self.meas_eval(xr, par, dx=True))))

    def simulate(self, steps, mc_sims=1):
        """State-space model simulation.

        SSM simulation starting from initial conditions for a given number of time steps

        Parameters
        ----------
        steps : int
            Number of time steps in state trajectory
        mc_sims : int
            Number of trajectories to simulate (the initial state is drawn randomly)

        Returns
        -------
        tuple
            Tuple (x, z) where both element are of type numpy.ndarray and where:

                * x : 3-D array of shape (self.xD, steps, mc_sims) containing the true system state trajectory
                * z : 3-D array of shape (self.zD, steps, mc_sims) containing simulated measurements of the system state
        """

        # allocate space for state and measurement sequences
        x = np.zeros((self.xD, steps, mc_sims))
        z = np.zeros((self.zD, steps, mc_sims))

        # generate state and measurement noise
        q = self.state_noise_sample((mc_sims, steps))
        r = self.measurement_noise_sample((mc_sims, steps))

        # generate initial conditions, store initial states at k=0
        x0 = self.initial_condition_sample(mc_sims)  # (D, mc_sims)
        x[:, 0, :] = x0

        # simulate SSM `mc_sims` times for `steps` time steps
        for imc in range(mc_sims):
            for k in range(1, steps):
                theta = self.par_fcn(k - 1)
                x[:, k, imc] = self.dyn_fcn(x[:, k-1, imc], q[:, k-1, imc], theta)
                z[:, k, imc] = self.meas_fcn(x[:, k, imc], r[:, k, imc], theta)
        return x, z

    def set_pars(self, key, value):
        self.pars[key] = value

    def get_pars(self, *keys):
        values = []
        for k in keys:
            values.append(self.pars.get(k))
        return values


class GaussianStateSpaceModel(StateSpaceModel):
    """
    State-space model with Gaussian noise and initial conditions.
    """

    def __init__(self, x0_mean=None, x0_cov=None, q_mean=None, q_cov=None, r_mean=None, r_cov=None, q_gain=None):

        # use default value of statistics for Gaussian SSM if None provided
        kwargs = {
            'x0_mean': x0_mean if x0_mean is not None else np.zeros(self.xD),
            'x0_cov': x0_cov if x0_cov is not None else np.eye(self.xD),
            'q_mean': q_mean if q_mean is not None else np.zeros(self.qD),
            'q_cov': q_cov if q_cov is not None else np.eye(self.qD),
            'r_mean': r_mean if r_mean is not None else np.zeros(self.rD),
            'r_cov': r_cov if r_cov is not None else np.eye(self.rD),
            'q_gain': q_gain if q_gain is not None else np.eye(self.qD)
        }
        super(GaussianStateSpaceModel, self).__init__(**kwargs)

    @abstractmethod
    def dyn_fcn(self, x, q, pars):
        """ System dynamics.

        Abstract method for the system dynamics.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like
            Parameters of the system dynamics

        Returns
        -------
        1-D numpy.ndarray of shape (self.xD,)
            system state in the next time step
        """
        pass

    @abstractmethod
    def meas_fcn(self, x, r, pars):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            system state
        r : 1-D array_like of shape (self.rD,)
            measurement noise
        pars : 1-D array_like
            parameters of the measurement model

        Returns
        -------
        1-D numpy.ndarray of shape (self.zD,)
            measurement of the state
        """
        pass

    @abstractmethod
    def par_fcn(self, time):
        """Parameter function of the system dynamics and measurement model.

        Abstract method for the parameter function of the whole state-space model. The implementation should ensure
        that the system dynamics parameters come before the measurement model parameters in the returned vector of
        parameters.

        Parameters
        ----------
        time : int
            Discrete time step

        Returns
        -------
        1-D numpy.ndarray of shape (self.pD,)
            Vector of parameters at a given time.
        """
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, q, pars):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the system dynamics, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.qD)
        """
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, pars):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        r : 1-D array_like of shape (self.qD,)
            Measurement noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.rD)
        """
        pass

    def state_noise_sample(self, size=None):
        q_mean, q_cov = self.get_pars('q_mean', 'q_cov')
        return np.random.multivariate_normal(q_mean, q_cov, size).T

    def measurement_noise_sample(self, size=None):
        r_mean, r_cov = self.get_pars('r_mean', 'r_cov')
        return np.random.multivariate_normal(r_mean, r_cov, size).T

    def initial_condition_sample(self, size=None):
        x0_mean, x0_cov = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(x0_mean, x0_cov, size).T


class StudentStateSpaceModel(StateSpaceModel):

    def __init__(self, x0_mean=None, x0_cov=None, x0_dof=None, q_mean=None, q_cov=None, q_dof=None, q_gain=None,
                 r_mean=None, r_cov=None, r_dof=None):
        """
        State-space model where the noises are Student distributed.
        Takes covariances instead of scale matrices.

        Parameters
        ----------
        x0_mean
        x0_cov
        x0_dof
        q_mean
        q_cov
        q_dof
        q_gain
        r_mean
        r_cov
        r_dof
        """
        kwargs = {
            'x0_mean': x0_mean if x0_mean is not None else np.zeros(self.xD),
            'x0_cov': x0_cov if x0_cov is not None else np.eye(self.xD),
            'x0_dof': x0_dof if x0_dof is not None else 4.0,  # desired DOF
            'q_mean': q_mean if q_mean is not None else np.zeros(self.qD),
            'q_cov': q_cov if q_cov is not None else np.eye(self.qD),
            'q_gain': q_gain if q_gain is not None else np.eye(self.qD),
            'q_dof': q_dof if q_dof is not None else 4.0,
            'r_mean': r_mean if r_mean is not None else np.zeros(self.rD),
            'r_cov': r_cov if r_cov is not None else np.eye(self.rD),
            'r_dof': r_dof if r_dof is not None else 4.0,
        }
        super(StudentStateSpaceModel, self).__init__(**kwargs)

    @abstractmethod
    def dyn_fcn(self, x, q, pars):
        """ System dynamics.

        Abstract method for the system dynamics.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like
            Parameters of the system dynamics

        Returns
        -------
        1-D numpy.ndarray of shape (self.xD,)
            system state in the next time step
        """
        pass

    @abstractmethod
    def meas_fcn(self, x, r, pars):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            system state
        r : 1-D array_like of shape (self.rD,)
            measurement noise
        pars : 1-D array_like
            parameters of the measurement model

        Returns
        -------
        1-D numpy.ndarray of shape (self.zD,)
            measurement of the state
        """
        pass

    @abstractmethod
    def par_fcn(self, time):
        """Parameter function of the system dynamics and measurement model.

        Abstract method for the parameter function of the whole state-space model. The implementation should ensure
        that the system dynamics parameters come before the measurement model parameters in the returned vector of
        parameters.

        Parameters
        ----------
        time : int
            Discrete time step

        Returns
        -------
        1-D numpy.ndarray of shape (self.pD,)
            Vector of parameters at a given time.
        """
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, q, pars):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the system dynamics, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.qD)
        """
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, pars):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        r : 1-D array_like of shape (self.qD,)
            Measurement noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.rD)
        """
        pass

    def state_noise_sample(self, size=None):
        q_mean, q_cov, q_dof = self.get_pars('q_mean', 'q_cov', 'q_dof')
        return multivariate_t(q_mean, q_cov, q_dof, size).T

    def measurement_noise_sample(self, size=None):
        r_mean, r_cov, r_dof = self.get_pars('r_mean', 'r_cov', 'r_dof')
        return multivariate_t(r_mean, r_cov, r_dof, size).T

    def initial_condition_sample(self, size=None):
        x0_mean, x0_cov, x0_dof = self.get_pars('x0_mean', 'x0_cov', 'x0_dof')
        return multivariate_t(x0_mean, x0_cov, x0_dof, size).T


class UNGMSys(StateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1

    q_additive = True
    r_additive = True

    def __init__(self):
        pars = {
            'x0_mean': np.atleast_1d(0.0),
            'x0_cov': np.atleast_2d(5.0),
            'q_mean_0': np.zeros(self.qD),
            'q_mean_1': np.zeros(self.qD),
            'q_cov_0': 10 * np.eye(self.qD),
            'q_cov_1': 100 * np.eye(self.qD),
            'r_mean_0': np.zeros(self.rD),
            'r_mean_1': np.zeros(self.rD),
            'r_cov_0': 0.01 * np.eye(self.rD),
            'r_cov_1': 1 * np.eye(self.rD),
        }
        super(UNGMSys, self).__init__(**pars)

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * x[0] ** 2]) + r

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * pars[0])])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * r[0] * x[0], 0.05 * x[0] ** 2])

    def initial_condition_sample(self, size=None):
        m, c = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(m, c, size).T

    def state_noise_sample(self, size=None):
        m0, c0 = self.get_pars('q_mean_0', 'q_cov_0')
        m1, c1 = self.get_pars('q_mean_1', 'q_cov_1')

        # samples from 2-component Gaussian mixture
        return bigauss_mixture(m0, c0, m1, c1, 0.8, size)

    def measurement_noise_sample(self, size=None):
        m0, c0 = self.get_pars('r_mean_0', 'r_cov_0')
        m1, c1 = self.get_pars('r_mean_1', 'r_cov_1')

        return bigauss_mixture(m0, c0, m1, c1, 0.8, size)


class UNGM(StudentStateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1

    q_additive = True
    r_additive = True

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        super(UNGM, self).__init__(**kwargs)
        kwargs = {
            'x0_mean': np.atleast_1d(x0_mean),
            'x0_cov': np.atleast_2d(x0_cov),
            'x0_dof': 4.0,
            'q_mean': np.atleast_1d(q_mean),
            'q_cov': np.atleast_2d(q_cov),
            'q_dof': 4.0,
            'r_mean': np.atleast_1d(r_mean),
            'r_cov': np.atleast_2d(r_cov),
            'r_dof': 4.0,
        }
        super(UNGM, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * x[0] ** 2]) + r

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * pars[0])])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * r[0] * x[0], 0.05 * x[0] ** 2])


def ungm_demo(steps=250, mc_sims=100):
    sys = UNGMSys()
    x, z = sys.simulate(steps, mc_sims)

    # SSM noise covariances should follow the system
    ssm = UNGM(x0_mean=1.0, q_cov=10.0, r_cov=0.01)

    # kernel parameters for TPQ and GPQ filters
    # TPQ Student
    # par_dyn_tp = np.array([[1.8, 3.0]])
    # par_obs_tp = np.array([[0.4, 1.0, 1.0]])
    par_dyn_tp = np.array([[3.0, 1.0]])
    par_obs_tp = np.array([[3.0, 3.0]])
    # GPQ Student
    par_dyn_gpqs = par_dyn_tp
    par_obs_gpqs = par_obs_tp
    # GPQ Kalman
    par_dyn_gpqk = np.array([[1.0, 0.5]])
    par_obs_gpqk = np.array([[1.0, 1, 10]])
    # parameters of the point-set
    kappa = 0.0
    par_pt = {'kappa': kappa}

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        UnscentedKalman(ssm, kappa=kappa),
        FSQStudent(ssm, kappa=kappa, dof=3.0),
        # FSQStudent(ssm, kappa=kappa, dof=4.0),
        # FSQStudent(ssm, kappa=kappa, dof=8.0),
        # FSQStudent(ssm, kappa=kappa, dof=100.0),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=3.0, point_par=par_pt),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=4.0, point_par=par_pt),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=10.0, point_par=par_pt),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=100.0, point_par=par_pt),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=500.0, point_par=par_pt),
        GPQStudent(ssm, par_dyn_gpqs, par_obs_gpqs, dof=4.0, point_hyp=par_pt),
        # TPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
        # GPQKalman(ssm, par_dyn_tp, par_obs_tp, point_hyp=par_pt),
        # GPQMOKalman(ssm, par_dyn_tp, par_obs_tp, point_par=par_pt),
        # TPQMOStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=10.0, point_par=par_pt),
    )
    itpq = np.argwhere([isinstance(filters[i], TPQStudent) for i in range(len(filters))]).squeeze(axis=1)[0]

    # assign weights approximated by MC with lots of samples
    # very dirty code
    pts = filters[itpq].tf_dyn.model.points
    kern = filters[itpq].tf_dyn.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_dyn, BQTransform):
            f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
            f.tf_dyn.Q = Q
    pts = filters[itpq].tf_meas.model.points
    kern = filters[itpq].tf_meas.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_meas, BQTransform):
            f.tf_meas.wm, f.tf_meas.Wc, f.tf_meas.Wcc = wm, wc, wcc
            f.tf_meas.Q = Q

    # print kernel parameters
    import pandas as pd
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(x.shape[0])]
    partable = pd.DataFrame(np.vstack((par_dyn_tp, par_obs_tp)), columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)

    # run all filters
    mf, Pf = run_filters(filters, z)

    # compute average RMSE and INC from filtered trajectories
    rmse_avg, lcr_avg = eval_perf_scores(x, mf, Pf)

    # variance of average metrics
    from ssmtoybox.utils import bootstrap_var
    var_rmse_avg = np.zeros((len(filters),))
    var_lcr_avg = np.zeros((len(filters),))
    for fi in range(len(filters)):
        var_rmse_avg[fi] = bootstrap_var(rmse_avg[:, fi], int(1e4))
        var_lcr_avg[fi] = bootstrap_var(lcr_avg[:, fi], int(1e4))

    # save trajectories, measurements and metrics to file for later processing (tables, plots)
    data_dict = {
        'x': x,
        'z': z,
        'mf': mf,
        'Pf': Pf,
        'rmse_avg': rmse_avg,
        'lcr_avg': lcr_avg,
        'var_rmse_avg': var_rmse_avg,
        'var_lcr_avg': var_lcr_avg,
        'steps': steps,
        'mc_sims': mc_sims,
        'par_dyn_tp': par_dyn_tp,
        'par_obs_tp': par_obs_tp,
    }
    joblib.dump(data_dict, f'ungm_simdata_{steps}k_{mc_sims}mc.dat')

    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'STD(MEAN_RMSE)', 'MEAN_INC', 'STD(MEAN_INC)']
    data = np.array([rmse_avg.mean(axis=0), np.sqrt(var_rmse_avg), lcr_avg.mean(axis=0), np.sqrt(var_lcr_avg)]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)


def ungm_plots_tables(datafile):

    # extract true/filtered state trajectories, measurements and evaluated metrics from *.mat data file
    d = joblib.load(datafile)
    x, z, mf, Pf = d['x'], d['z'], d['mf'], d['Pf']
    rmse_avg, lcr_avg = d['rmse_avg'], d['lcr_avg']
    var_rmse_avg, var_lcr_avg = d['var_rmse_avg'].squeeze(), d['var_lcr_avg'].squeeze()
    steps, mc_sims = d['steps'], d['mc_sims']

    # TABLES
    import pandas as pd

    # limit display of decimal places
    pd.set_option('display.precision', 4)

    # filter/metric labels
    f_label = ['UKF', 'SF', r'TPQSF($\nu$=3)', r'TPQSF($\nu$=4)',
               r'TPQSF($\nu$=10)', r'TPQSF($\nu$=100)', r'TPQSF($\nu$=500)', 'GPQSF']
    m_label = ['MEAN_RMSE', 'VAR(MEAN_RMSE)', 'MEAN_INC', 'VAR(MEAN_INC)']

    # form data array, put in DataFrame and print
    data = np.array([rmse_avg.mean(axis=0), var_rmse_avg, lcr_avg.mean(axis=0), var_lcr_avg]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # save table to latex
    with open('ungm_rmse_inc.tex', 'w') as f:
        table.to_latex(f)

    # plots
    fp = FigurePrint()

    # RMSE and INC box plots
    fig, ax = plt.subplots()
    ax.boxplot(rmse_avg)
    ax.set_ylabel('Average RMSE')
    ax.set_ylim(0, 80)
    xtickNames = plt.setp(ax, xticklabels=f_label)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.tight_layout(pad=0.1)
    fp.savefig('ungm_rmse_boxplot')

    fig, ax = plt.subplots()
    ax.boxplot(lcr_avg)
    ax.set_ylabel('Average INC')
    xtickNames = plt.setp(ax, xticklabels=f_label)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.tight_layout(pad=0.1)
    fp.savefig('ungm_inc_boxplot')

    # filtered state and covariance
    # fig, ax = plt.subplots(3, 1, sharex=True)
    # time = np.arange(1, steps + 1)
    # for fi, f in enumerate(filters):
    #     # true state
    #     ax[fi].plot(time, x[0, :, 0], 'r--', alpha=0.5)
    #
    #     # measurements
    #     ax[fi].plot(time, z[0, :, 0], 'k.')
    #
    #     xhat = mf[0, :, 0, fi]
    #     std = np.sqrt(Pf[0, 0, :, 0, fi])
    #     ax[fi].plot(time, xhat, label=f.__class__.__name__)
    #     ax[fi].fill_between(time, xhat - 2 * std, xhat + 2 * std, alpha=0.15)
    #     ax[fi].axis([None, None, -50, 50])
    #     ax[fi].legend()
    # plt.show()
    #
    # # compare posterior variances with outliers
    # plt.figure()
    # plt.plot(time, z[0, :, 0], 'k.')
    # for fi, f in enumerate(filters):
    #     plt.plot(time, 2 * np.sqrt(Pf[0, 0, :, 0, fi]), label=f.__class__.__name__)
    # plt.legend()
    # plt.show()


class ConstantVelocityRadarSys(StateSpaceModel):
    """
    See: Arasaratnam et al.: Discrete-Time Nonlinear Filtering Algorithms Using Gauss–Hermite Quadrature
    """
    xD = 4
    zD = 2
    qD = 2
    rD = 2

    q_additive = True
    r_additive = True

    def __init__(self, dt=0.5):
        self.dt = dt
        self.q_gain = np.array([[dt**2/2, 0],
                                [dt, 0],
                                [0, dt**2/2],
                                [0, dt]])
        pars = {
            'x0_mean': np.array([10000, 300, 1000, -40]),  # m, m/s, m, m/s
            'x0_cov': np.diag([100**2, 10**2, 100**2, 10**2]),
            'q_mean': np.zeros((self.qD, )),
            'q_cov': np.diag([50, 5]),  # m^2/s^4, m^2/s^4
            'q_gain': self.q_gain,
            'r_mean_0': np.zeros((self.rD, )),
            # 'r_cov_0': np.diag([50, 0.4]),  # m^2, mrad^2
            'r_cov_0': np.diag([50, 0.4e-6]),  # m^2, rad^2
            'r_mean_1': np.zeros((self.rD,)),
            # 'r_cov_1': np.diag([5000, 16]),  # m^2, mrad^2
            'r_cov_1': np.diag([5000, 1.6e-5]),  # m^2, rad^2
        }
        super(ConstantVelocityRadarSys, self).__init__(**pars)

    def dyn_fcn(self, x, q, pars):
        A = np.array([[1, self.dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, self.dt],
                      [0, 0, 0, 1]])
        return A.dot(x) + self.q_gain.dot(q)

    def meas_fcn(self, x, r, pars):
        rang = np.sqrt(x[0]**2 + x[2]**2)
        theta = np.arctan2(x[2], x[0])
        return np.array([rang, theta]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def initial_condition_sample(self, size=None):
        m0, c0 = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(m0, c0, size).T

    def state_noise_sample(self, size=None):
        m0, c0 = self.get_pars('q_mean', 'q_cov')
        return np.random.multivariate_normal(m0, c0, size).T

    def measurement_noise_sample(self, size=None):
        m0, c0, m1, c1 = self.get_pars('r_mean_0', 'r_cov_0', 'r_mean_1', 'r_cov_1')
        return bigauss_mixture(m0, c0, m1, c1, 0.85, size)


class ConstantVelocityRadar(StudentStateSpaceModel):
    """
    See: Kotecha, Djuric, Gaussian Particle Filter
    """
    xD = 4
    zD = 2
    qD = 2
    rD = 2

    q_additive = True
    r_additive = True

    def __init__(self, dt=0.5):
        self.dt = dt
        self.q_gain = np.array([[dt ** 2 / 2, 0],
                                [dt, 0],
                                [0, dt ** 2 / 2],
                                [0, dt]])
        pars = {
            'x0_mean': np.array([10175, 295, 980, -35]),  # m, m/s, m, m/s
            'x0_cov': np.diag([100**2, 10**2, 100**2, 10**2]),
            'x0_dof': 1000.0,
            'q_mean': np.zeros((self.qD, )),
            'q_cov': np.diag([50, 5]),  # m^2/s^4, m^2/s^4
            'q_dof': 1000.0,
            'q_gain': self.q_gain,
            'r_mean': np.zeros((self.rD, )),
            # 'r_cov': np.diag([50, 0.4]),  # m^2, mrad^2
            'r_cov': np.diag([50, 0.4e-6]),  # m^2, rad^2
            'r_dof': 4.0,
        }
        super(ConstantVelocityRadar, self).__init__(**pars)

    def dyn_fcn(self, x, q, pars):
        A = np.array([[1, self.dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, self.dt],
                      [0, 0, 0, 1]])
        return A.dot(x) + self.q_gain.dot(q)

    def meas_fcn(self, x, r, pars):
        rang = np.sqrt(x[0]**2 + x[2]**2)
        theta = np.arctan2(x[2], x[0])
        return np.array([rang, theta]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


def constant_velocity_radar_demo(steps=100, mc_sims=1000):
    print('Constant Velocity Radar Tracking with Glint Noise')
    print('K = {:d}, MC = {:d}'.format(steps, mc_sims))

    sys = ConstantVelocityRadarSys()
    x, z = sys.simulate(steps, mc_sims)

    # import matplotlib.pyplot as plt
    # for i in range(mc_sims):
    #     plt.plot(x[0, :, i], x[2, :, i], 'b', alpha=0.15)
    # plt.show()

    # SSM noise covariances should follow the system
    ssm = ConstantVelocityRadar()

    # kernel parameters for TPQ and GPQ filters
    # TPQ Student
    par_dyn_tp = np.array([[1, 100, 100, 100, 100]], dtype=float)
    par_obs_tp = np.array([[0.05, 10, 100, 10, 100]], dtype=float)
    # parameters of the point-set
    kappa = 0.0
    par_pt = {'kappa': kappa}

    # print kernel parameters
    import pandas as pd
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(x.shape[0])]
    partable = pd.DataFrame(np.vstack((par_dyn_tp, par_obs_tp)), columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        UnscentedKalman(ssm, kappa=kappa),
        FSQStudent(ssm, kappa=kappa, dof=4.0),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=2.2, point_par=par_pt),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=4.0, point_par=par_pt),
        GPQStudent(ssm, par_dyn_tp, par_obs_tp, dof=4.0, point_hyp=par_pt),
    )

    # assign weights approximated by MC with lots of samples
    # very dirty code
    itpq = np.argwhere([isinstance(filters[i], TPQStudent) for i in range(len(filters))]).squeeze(axis=1)[0]
    pts = filters[itpq].tf_dyn.model.points
    kern = filters[itpq].tf_dyn.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    # weights_data = joblib.load('tpq_weights.dat')
    # wm, wc, wcc, Q = tuple(weights_data['dyn'].values())
    for f in filters:
        if isinstance(f.tf_dyn, BQTransform):
            f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
            f.tf_dyn.Q = Q
    pts = filters[itpq].tf_meas.model.points
    kern = filters[itpq].tf_meas.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    # wm, wc, wcc, Q = tuple(weights_data['obs'].values())
    for f in filters:
        if isinstance(f.tf_meas, BQTransform):
            f.tf_meas.wm, f.tf_meas.Wc, f.tf_meas.Wcc = wm, wc, wcc
            f.tf_meas.Q = Q

    # run all filters
    mf, Pf = run_filters(filters, z)

    # evaluate scores
    pos_x, pos_mf, pos_Pf = x[[0, 2], ...], mf[[0, 2], ...], Pf[np.ix_([0, 2], [0, 2])]
    vel_x, vel_mf, vel_Pf = x[[1, 3], ...], mf[[1, 3], ...], Pf[np.ix_([1, 3], [1, 3])]
    pos_rmse, pos_lcr = eval_perf_scores(pos_x, pos_mf, pos_Pf)
    vel_rmse, vel_lcr = eval_perf_scores(vel_x, vel_mf, vel_Pf)
    rmse_avg, lcr_avg = eval_perf_scores(x, mf, Pf)

    # variance of average metrics
    from ssmtoybox.utils import bootstrap_var
    var_rmse_avg = np.zeros((len(filters),))
    var_lcr_avg = np.zeros((len(filters),))
    for fi in range(len(filters)):
        var_rmse_avg[fi] = bootstrap_var(rmse_avg[:, fi], int(1e4))
        var_lcr_avg[fi] = bootstrap_var(lcr_avg[:, fi], int(1e4))

    # save trajectories, measurements and metrics to file for later processing (tables, plots)
    data_dict = {
        'x': x,
        'z': z,
        'mf': mf,
        'Pf': Pf,
        'rmse_avg': rmse_avg,
        'lcr_avg': lcr_avg,
        'var_rmse_avg': var_rmse_avg,
        'var_lcr_avg': var_lcr_avg,
        'pos_rmse': pos_rmse,
        'pos_lcr': pos_lcr,
        'vel_rmse': vel_rmse,
        'vel_lcr': vel_lcr,
        'steps': steps,
        'mc_sims': mc_sims,
        'par_dyn_tp': par_dyn_tp,
        'par_obs_tp': par_obs_tp,
        'f_label': [f.__class__.__name__ for f in filters]  # ['UKF', 'SF', r'TPQSF($\nu$=20)', 'GPQSF']
    }
    joblib.dump(data_dict, f'cv_radar_simdata_{steps}k_{mc_sims}mc.dat')

    # print out table
    # mean overall RMSE and INC with bootstrapped variances
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'STD(MEAN_RMSE)', 'MEAN_INC', 'STD(MEAN_INC)']
    data = np.array([rmse_avg.mean(axis=0), np.sqrt(var_rmse_avg), lcr_avg.mean(axis=0), np.sqrt(var_lcr_avg)]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # mean/max RMSE and INC
    m_label = ['MEAN_RMSE', 'MAX_RMSE', 'MEAN_INC', 'MAX_INC']
    pos_data = np.array([pos_rmse.mean(axis=0), pos_rmse.max(axis=0), pos_lcr.mean(axis=0), pos_lcr.max(axis=0)]).T
    vel_data = np.array([vel_rmse.mean(axis=0), vel_rmse.max(axis=0), vel_lcr.mean(axis=0), vel_lcr.max(axis=0)]).T
    pos_table = pd.DataFrame(pos_data, f_label, m_label)
    pos_table.index.name = 'Position'
    vel_table = pd.DataFrame(vel_data, f_label, m_label)
    vel_table.index.name = 'Velocity'
    print(pos_table)
    print(vel_table)

    # plot metrics
    import matplotlib.pyplot as plt
    time = np.arange(1, steps + 1)
    fig, ax = plt.subplots(2, 1, sharex=True)
    for fi, f in enumerate(filters):
        ax[0].semilogy(time, pos_rmse[..., fi], label=f.__class__.__name__)
        ax[1].semilogy(time, vel_rmse[..., fi], label=f.__class__.__name__)
    plt.legend()
    plt.show()


def constant_velocity_radar_plots_tables(datafile):

    # extract true/filtered state trajectories, measurements and evaluated metrics from *.mat data file
    d = joblib.load(datafile)
    # x, z, mf, Pf = d['x'], d['z'], d['mf'], d['Pf']
    rmse_avg, lcr_avg = d['rmse_avg'], d['lcr_avg']
    var_rmse_avg, var_lcr_avg = d['var_rmse_avg'].squeeze(), d['var_lcr_avg'].squeeze()
    pos_rmse, pos_lcr = d['pos_rmse'], d['pos_lcr']
    vel_rmse, vel_lcr = d['vel_rmse'], d['vel_lcr']
    steps, mc_sims = d['steps'], d['mc_sims']

    # TABLES
    import pandas as pd

    # limit display of decimal places
    pd.set_option('display.precision', 4)

    # filter/metric labels
    f_label = d['f_label']
    # f_label = ['UKF', 'SF', 'TPQSF\n' + r'$(\nu_g=4)$', 'TPQSF\n' + r'$(\nu_g=10)$',
    #            'TPQSF\n' + r'$(\nu_g=20)$', 'GPQSF']
    m_label = ['MEAN_RMSE', 'VAR(MEAN_RMSE)', 'MEAN_INC', 'VAR(MEAN_INC)']

    # form data array, put in DataFrame and print
    data = np.array([rmse_avg.mean(axis=0), var_rmse_avg, lcr_avg.mean(axis=0), var_lcr_avg]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # save table to latex
    with open('cv_radar_rmse_inc.tex', 'w') as f:
        table.to_latex(f)

    # plots
    # import matplotlib.pyplot as plt
    # from fusion_paper.figprint import FigurePrint
    fp = FigurePrint()

    # position and velocity RMSE plots
    time = np.arange(1, steps+1)
    fig, ax = plt.subplots(2, 1, sharex=True)

    for fi, f in enumerate(f_label):
        ax[0].semilogy(time, pos_rmse[..., fi], label=f)
        ax[1].semilogy(time, vel_rmse[..., fi], label=f)
    ax[0].set_ylabel('Position')
    ax[1].set_ylabel('Velocity')
    ax[1].set_xlabel('time step [k]')
    plt.legend()
    plt.tight_layout(pad=0)
    fp.savefig('cv_radar_rmse_semilogy')

    # RMSE and INC box plots
    fig, ax = plt.subplots()
    ax.boxplot(rmse_avg)
    ax.set_ylabel('Average RMSE')
    # ax.set_ylim(0, 80)
    xtickNames = plt.setp(ax, xticklabels=f_label)
    # plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.tight_layout(pad=0.1)
    fp.savefig('cv_radar_rmse_boxplot')

    fig, ax = plt.subplots()
    ax.boxplot(lcr_avg)
    ax.set_ylabel('Average INC')
    xtickNames = plt.setp(ax, xticklabels=f_label)
    # plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.tight_layout(pad=0.1)
    fp.savefig('cv_radar_inc_boxplot')


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    np.random.seed(42)

    steps, mc_sims = 100, 1000
    constant_velocity_radar_demo(steps, mc_sims)
    constant_velocity_radar_plots_tables(f'cv_radar_simdata_{steps}k_{mc_sims}mc.dat')

    # steps, mc_sims = 250, 500
    # ungm_demo(steps, mc_sims)
    # ungm_plots_tables(f'ungm_simdata_{steps}k_{mc_sims}mc.dat')
