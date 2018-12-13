from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.linalg as la
from scipy.linalg import cho_factor, cho_solve, block_diag
from scipy.stats import multivariate_normal
from numpy import newaxis as na
from ssmtoybox.ssmod import TransitionModel, MeasurementModel
from ssmtoybox.bq.bqmtran import GaussianProcessTransform, MultiOutputGaussianProcessTransform, StudentTProcessTransform, MultiOutputStudentTProcessTransform, BayesSardTransform
from ssmtoybox.mtran import MomentTransform, LinearizationTransform, TaylorGPQDTransform, \
    SphericalRadialTransform, UnscentedTransform, GaussHermiteTransform, FullySymmetricStudentTransform, \
    TruncatedSphericalRadialTransform, TruncatedUnscentedTransform, TruncatedGaussHermiteTransform
from ssmtoybox.utils import StudentRV
import warnings


class StateSpaceInference(metaclass=ABCMeta):
    """
    Base class for all local state-space inference algorithms, including nonlinear filters and smoothers.

    Parameters
    ----------
    mod_dyn : TransitionModel
        State transition model defining system dynamics.

    mod_obs : MeasurementModel
        Measurement model describing the measurement formation process.

    tf_dyn : MomentTransform
        Moment transform for computing predictive state moments.

    tf_obs : MomentTransform
        Moment transform for computing predictive measurement moments.

    """

    def __init__(self, mod_dyn, mod_obs, tf_dyn, tf_obs):

        # state-space model of a dynamical system whose state is to be estimated
        assert isinstance(mod_dyn, TransitionModel) and isinstance(mod_obs, MeasurementModel)
        self.mod_dyn = mod_dyn
        self.mod_obs = mod_obs

        # separate moment transforms for system dynamics and measurement model
        assert isinstance(tf_dyn, MomentTransform) and isinstance(tf_obs, MomentTransform)
        self.tf_dyn = tf_dyn
        self.tf_obs = tf_obs

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
        """
        Forward pass of the state-space inference (filtering).

        Parameters
        ----------
        data : (dim, n_time_steps) ndarray
            Measurements to process

        Returns
        -------
        filtered_mean : (dim, n_time_steps) ndarray
            Filtered mean of the system state.

        filtered_cov : (dim, dim, n_time_steps) ndarray
            Filtered covariance of the system state.

        """

        self.D, self.N = data.shape
        self.fi_mean = np.zeros((self.mod_dyn.dim_state, self.N+1))
        self.fi_cov = np.zeros((self.mod_dyn.dim_state, self.mod_dyn.dim_state, self.N+1))
        # FIXME: why save x0 to fi_mean, fi_cov when they get trimmed off in the end?
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
        """
        Backward pass of the state-space inference (smoothing).

        Returns
        -------
        smoothed_mean : ndarray
            Smoothed mean of the system state.

        smoothed_cov : ndarray
            Smoothed covariance of the system state.

        """

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
        return self.sm_mean[:, 1:, ...], self.sm_cov[:, :, 1:, ...]

    def reset(self):
        """Reset internal variables and flags."""
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
        """
        Abstract method for time update, which computes predictive moments of state and measurement.

        Parameters
        ----------
        time : int
            Time step. Importannt for t-variant systems.

        theta_dyn :
            Parameters of the moment transform computing the predictive state moments.

        theta_obs :
            Parameters of the moment transform computing the predictive measurement moments.

        Returns
        -------

        """
        pass

    @abstractmethod
    def _measurement_update(self, y, time=None):
        """
        Abstract method for measurement update, which takes predictive state and measurement moments and produces
        filtered state mean and covariance.

        Parameters
        ----------
        y : (dim, ) ndarray
            Measurement vector.

        time : int
            Time step. Important for t-variant systems.

        Returns
        -------

        """
        pass

    @abstractmethod
    def _smoothing_update(self):
        """
        Abstract method for smoothing update, which takes filtered states and predictive states from the forward pass
        and goes backward in time producing smoothed moments of the system state.

        Returns
        -------

        """
        pass


class GaussianInference(StateSpaceInference):
    """
    Base class for all Gaussian state-space inference algorithms, such as nonlinear Kalman filters and smoothers.

    dyn : TransitionModel
        Transition model defining system dynamics with Gaussian distributed noises and initial conditions.

    obs : MeasurementModel
        Measurement model with Gaussian distributed noise.

    tf_dyn : MomentTransform
        Moment transform for the dynamics. Computes predictive state mean, covariance and cross-covariance.

    tf_obs : MomentTransform
        Moment transform for the measurement model. Computes predictive measurement mean,
        covariance and cross-covariance.
    """

    def __init__(self, mod_dyn, mod_obs, tf_dyn, tf_obs):

        # dynamical system whose state is to be estimated
        assert isinstance(mod_dyn, TransitionModel) and isinstance(mod_obs, MeasurementModel)

        # set initial condition mean and covariance, and noise covariances
        self.x0_mean, self.x0_cov = mod_dyn.init_rv.get_stats()
        self.q_mean, self.q_cov = mod_dyn.noise_rv.get_stats()
        self.r_mean, self.r_cov = mod_obs.noise_rv.get_stats()

        self.G = mod_dyn.noise_gain
        # initial moments are taken to be the first filtered estimate
        self.x_mean_fi, self.x_cov_fi = self.x0_mean, self.x0_cov

        super(GaussianInference, self).__init__(mod_dyn, mod_obs, tf_dyn, tf_obs)

    def reset(self):
        """Reset internal variables and flags."""
        self.x_mean_fi, self.x_cov_fi = self.x0_mean, self.x0_cov
        super(GaussianInference, self).reset()

    def _time_update(self, time, theta_dyn=None, theta_obs=None):
        """
        Time update for Gaussian filters and smoothers, computing predictive moments of state and measurement.

        Parameters
        ----------
        time : int
            Time step. Important for t-variant systems.

        theta_dyn : ndarray
            Parameters of the moment transform computing the predictive state moments.

        theta_obs : ndarray
            Parameters of the moment transform computing the predictive measurement moments.
        """

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_fi if self.mod_dyn.noise_additive else np.hstack((self.x_mean_fi, self.q_mean))
        cov = self.x_cov_fi if self.mod_dyn.noise_additive else block_diag(self.x_cov_fi, self.q_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute predicted state mean, covariance
        self.x_mean_pr, self.x_cov_pr, self.xx_cov = self.tf_dyn.apply(self.mod_dyn.dyn_eval, mean, cov,
                                                                       np.atleast_1d(time), theta_dyn)
        if self.mod_dyn.noise_additive:
            self.x_cov_pr += self.G.dot(self.q_cov).dot(self.G.T)

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_pr if self.mod_obs.noise_additive else np.hstack((self.x_mean_pr, self.r_mean))
        cov = self.x_cov_pr if self.mod_obs.noise_additive else block_diag(self.x_cov_pr, self.r_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute measurement mean, covariance
        self.y_mean_pr, self.y_cov_pr, self.xy_cov = self.tf_obs.apply(self.mod_obs.meas_eval, mean, cov,
                                                                       np.atleast_1d(time), theta_obs)
        # in additive case, noise covariances need to be added
        if self.mod_obs.noise_additive:
            self.y_cov_pr += self.r_cov

        # in non-additive case, cross-covariances must be trimmed (has no effect in additive case)
        self.xy_cov = self.xy_cov[:, :self.mod_dyn.dim_state]
        self.xx_cov = self.xx_cov[:, :self.mod_dyn.dim_state]

    def _measurement_update(self, y, time=None):
        """
        Measurement update for Gaussian filters, which takes predictive state and measurement moments and produces
        filtered state mean and covariance.

        Parameters
        ----------
        y : (dim, ) ndarray
            Measurement vector.

        time : int
            Time step. Important for t-variant systems.

        Notes
        -----
        Implements general Gaussian filter measurement update in the form

        .. math::
        \[
            G_k = P^{xy}_{k|k-1}(P^y_{k|k-1})^{-1}
            m^x_{k|k} = m^x_{k|k-1} + G_k (y_k - m^y_{k|k-1})
            P^x_{k|k} = P^x_{k|k-1} - G_k P^y_{k|k-1} G^T_k
        \]
        """
        gain = cho_solve(cho_factor(self.y_cov_pr), self.xy_cov).T
        self.x_mean_fi = self.x_mean_pr + gain.dot(y - self.y_mean_pr)
        self.x_cov_fi = self.x_cov_pr - gain.dot(self.y_cov_pr).dot(gain.T)

    def _smoothing_update(self):
        """
        Smoothing update, which takes filtered states and predictive states from the forward pass and goes backward
        in time producing moments of the smoothed system state.

        Notes
        -----
        Implements general Gaussian Rauch-Tung-Striebel smoothing update equations in the form

        .. math::
        \[
            D_{k+1} = P^{xx}_{k+1|K}(P^x{k+1|k})^{-1}
            m^x_{k|K} = m^x_{k|k} + D_{k+1} (m^x_{k+1|K} - m^x_{k+1|k})
            P^x_{k|K} = P^x_{k|k} + D_{k+1} (P^x_{k+1|K} - P^x_{k+1|k}) D^T_{k+1}
        \]

        """
        gain = cho_solve(cho_factor(self.x_cov_pr), self.xx_cov).T
        self.x_mean_sm = self.x_mean_fi + gain.dot(self.x_mean_sm - self.x_mean_pr)
        self.x_cov_sm = self.x_cov_fi + gain.dot(self.x_cov_sm - self.x_cov_pr).dot(gain.T)


class ExtendedKalman(GaussianInference):
    """
    Extended Kalman filter and smoother.

    For linear dynamics and measurement model this is a Kalman filter and Rauch-Tung-Striebel smoother.
    """

    def __init__(self, dyn, obs):
        tf = LinearizationTransform(dyn.dim_in)
        th = LinearizationTransform(obs.dim_in)
        super(ExtendedKalman, self).__init__(dyn, obs, tf, th)


class CubatureKalman(GaussianInference):
    """ Cubature Kalman filter and smoother. """

    def __init__(self, dyn, obs):
        tf = SphericalRadialTransform(dyn.dim_in)
        th = SphericalRadialTransform(obs.dim_in)
        super(CubatureKalman, self).__init__(dyn, obs, tf, th)


class UnscentedKalman(GaussianInference):
    """
    Unscented Kalman filter and smoother.

    Parameters
    ----------
    kappa : float or None, optional
        Controls spread of points around the mean. If `None`, `kappa=max(3-dim, 0)`

    alpha : float, optional
    beta : float, optional
        Parameters of the Unscented transform.
    """

    def __init__(self, dyn, obs, kappa=None, alpha=1.0, beta=2.0):
        tf = UnscentedTransform(dyn.dim_in, kappa=kappa, alpha=alpha, beta=beta)
        th = UnscentedTransform(obs.dim_in, kappa=kappa, alpha=alpha, beta=beta)
        super(UnscentedKalman, self).__init__(dyn, obs, tf, th)


class GaussHermiteKalman(GaussianInference):
    """
    Gauss-Hermite Kalman filter and smoother.

    Parameters
    ----------
    deg : int, optional
        Degree of the Gauss-Hermite integration rule. Determines the number of sigma-points.
    """

    def __init__(self, dyn, obs, deg=3):
        tf = GaussHermiteTransform(dyn.dim_in, degree=deg)
        th = GaussHermiteTransform(obs.dim_in, degree=deg)
        super(GaussHermiteKalman, self).__init__(dyn, obs, tf, th)


class GaussianProcessKalman(GaussianInference):
    """
    Gaussian process quadrature Kalman filter (GPQKF) and smoother (see [1]_).

    Parameters
    ----------
    kern_par_dyn : ndarray
        Kernel parameters for GPQ transformation of the state moments.

    kern_par_obs : ndarray
        Kernel parameters for GPQ transformation of the measurement moments.

    kernel : str {'rbf'}, optional
        Kernel (covariance function) of the internal Gaussian process regression model.

    points : str {'sr', 'ut', 'gh', 'fs'}, optional
        Sigma-point set:

        ``sr``
            Spherical-radial sigma-points (originally used in CKF).
        ``ut``
            Unscented transform sigma-points (originally used in UKF).
        ``gh``
            Gauss-Hermite sigma-points (originally used in GHKF).
        ``fs``
            Fully-symmetric sigma-points [3]_ (originally used in [2]_).

    point_hyp : dict, optional
        Hyper-parameters of the sigma-point set.

    References
    ----------
    .. [1] Prüher, J. and Straka, O. Gaussian Process Quadrature Moment Transform,
           IEEE Transactions on Automatic Control, 2017, Pre-print, 1-1

    .. [2] Tronarp, F. and Hostettler, R. and Särkkä, S. Sigma-point Filtering for Nonlinear Systems with Non-additive
           Heavy-tailed Noise, 19th International Conference on Information Fusion, 2016, 1859-1866

    .. [3] J. McNamee and F. Stenger, Construction of fully symmetric numerical integration formulas,
           Numerische Mathematik, vol. 10, pp. 327–344, 1967

    """

    def __init__(self, dyn, obs, kern_par_dyn, kern_par_obs, kernel='rbf', points='ut', point_hyp=None):
        t_dyn = GaussianProcessTransform(dyn.dim_in, dyn.dim_state, kern_par_dyn, kernel, points, point_hyp)
        t_obs = GaussianProcessTransform(obs.dim_in, obs.dim_out, kern_par_obs, kernel, points, point_hyp)
        super(GaussianProcessKalman, self).__init__(dyn, obs, t_dyn, t_obs)


class BayesSardKalman(GaussianInference):
    """
    Bayes-Sard quadrature Kalman filter (BSQKF) and smoother (see [1]_).

    Parameters
    ----------
    kern_par_dyn : ndarray
        Kernel parameters for GPQ transformation of the state moments.

    kern_par_obs : ndarray
        Kernel parameters for GPQ transformation of the measurement moments.

    mulind_dyn : int or ndarray, optional
    mulind_obs : int or ndarray, optional
        Multi-indices for dynamics and observation models.

        ``int``
            Equivalent to multi-index defining all monomials of total degree less then or equal to the supplied int.
        ``ndarray``
            Matrix, where columns are multi-indices defining the basis functions (monomials) of the polynomial space.

    points : str {'sr', 'ut', 'gh', 'fs'}, optional
        Sigma-point set:

        ``sr``
            Spherical-radial sigma-points (originally used in CKF).
        ``ut``
            Unscented transform sigma-points (originally used in UKF).
        ``gh``
            Gauss-Hermite sigma-points (originally used in GHKF).
        ``fs``
            Fully-symmetric sigma-points [3]_ (originally used in [2]_).

    point_hyp : dict, optional
        Hyper-parameters of the sigma-point set.

    References
    ----------
    Prüher, J., Karvonen, T., Oates, C. J., Straka, O. and Särkkä, S.
    Improved Calibration of Numerical Integration Error in Sigma-Point Filters, https://export.arxiv.org/abs/1811.11474
    """

    def __init__(self, dyn, obs, kern_par_dyn, kern_par_obs, mulind_dyn=2, mulind_obs=2, points='ut', point_hyp=None):
        t_dyn = BayesSardTransform(dyn.dim_in, dyn.dim_state, kern_par_dyn, mulind_dyn, points, point_hyp)
        t_obs = BayesSardTransform(obs.dim_in, obs.dim_out, kern_par_obs, mulind_obs, points, point_hyp)
        super(BayesSardKalman, self).__init__(dyn, obs, t_dyn, t_obs)


class StudentProcessKalman(GaussianInference):
    """
    Student's t-process quadrature Kalman filter (TPQKF) and smoother (see [1]_).

    Parameters
    ----------
    kern_par_dyn : ndarray
        Kernel parameters for TPQ transformation of the state moments.

    kern_par_obs : ndarray
        Kernel parameters for TPQ transformation of the measurement moments.

    kernel : str {'rbf'}, optional
        Kernel (covariance function) of the internal Student's t-process regression model.

    points : str {'sr', 'ut', 'gh', 'fs'}, optional
        Sigma-point set:

        ``sr``
            Spherical-radial sigma-points (originally used in CKF).
        ``ut``
            Unscented transform sigma-points (originally used in UKF).
        ``gh``
            Gauss-Hermite sigma-points (originally used in GHKF).
        ``fs``
            Fully-symmetric sigma-points [3]_ (originally used in [2]_).

    point_hyp : dict, optional
        Hyper-parameters of the sigma-point set.

    nu : float, optional
        Degrees of freedom of the Student's t-regression model.

    References
    ----------
    .. [1] Prüher, J.; Tronarp, F.; Karvonen, T.; Särkkä, S. and Straka, O. Student-t Process Quadratures for
           Filtering of Non-linear Systems with Heavy-tailed Noise, 20th International Conference on Information
           Fusion, 2017 , 1-8

    .. [2] Tronarp, F. and Hostettler, R. and Särkkä, S. Sigma-point Filtering for Nonlinear Systems with Non-additive
           Heavy-tailed Noise, 19th International Conference on Information Fusion, 2016, 1859-1866

    .. [3] J. McNamee and F. Stenger, Construction of fully symmetric numerical integration formulas,
           Numerische Mathematik, vol. 10, pp. 327–344, 1967
    """

    def __init__(self, dyn, obs, kern_par_dyn, kern_par_obs, kernel='rbf', points='ut', point_hyp=None, nu=3.0):
        t_dyn = StudentTProcessTransform(dyn.dim_in, 1, kern_par_dyn, kernel, points, point_hyp, nu=nu)
        t_obs = StudentTProcessTransform(obs.dim_in, 1, kern_par_obs, kernel, points, point_hyp, nu=nu)
        super(StudentProcessKalman, self).__init__(dyn, obs, t_dyn, t_obs)


class StudentianInference(StateSpaceInference):
    """
    Base class for state-space inference algorithms, which assume that the state and measurement variables are jointly
    Student's t-distributed.

    Parameters
    ----------
    mod_dyn : TransitionModel
        Transition model defining system dynamics with Student distributed noises and initial conditions.

    mod_obs : MeasurementModel
        Measurement model with Student distributed noise.

    tf_dyn : MomentTransform
        Moment transform for the dynamics. Computes predictive state mean, covariance and cross-covariance.

    tf_obs : MomentTransform
        Moment transform for the measurement model. Computes predictive measurement mean,
        covariance and cross-covariance.

    dof : float, optional
        Degree of freedom parameter of the filtered density.

    fixed_dof : bool, optional
        If `True`, DOF will be fixed for all time steps, which preserves the heavy-tailed behaviour of the filter.
        If `False`, DOF will be increasing after each measurement update, which means the heavy-tailed behaviour is
        not preserved and therefore converges to a Gaussian filter.

    Notes
    -----
    Even though Student's t distribution is not parametrized by the covariance matrix like the Gaussian,
    the filter still produces mean and covariance of the state.
    """

    def __init__(self, mod_dyn, mod_obs, tf_dyn, tf_obs, dof=4.0, fixed_dof=True):
        # make sure initial state and noises are Student RVs
        if not isinstance(mod_dyn.init_rv, StudentRV):
            ValueError("Initial state is not Student RV.")
        if not isinstance(mod_dyn.noise_rv, StudentRV):
            ValueError("Process noise is not Student RV.")
        if not isinstance(mod_obs.noise_rv, StudentRV):
            ValueError("Measurement noise is not Student RV.")
        if dof <= 2.0:
            dof = 4.0
            warnings.warn("You supplied invalid DoF (must be > 2). Setting to dof=4.")

        # extract SSM parameters  # TODO get_stats() returns scale mat., convert it to cov. mat.
        self.x0_mean, self.x0_cov, self.x0_dof = mod_dyn.init_rv.get_stats()
        # self.x0_cov = (self.x0_dof/(self.x0_dof-2)) * self.x0_cov
        # initial filtered statistics are the initial state statistics
        self.x_mean_fi, self.x_cov_fi, self.dof_fi = self.x0_mean, self.x0_cov, self.x0_dof

        # state noise statistics
        self.q_mean, self.q_cov, self.q_dof = mod_dyn.noise_rv.get_stats()
        self.q_gain = mod_dyn.noise_gain

        # measurement noise statistics
        self.r_mean, self.r_cov, self.r_dof = mod_obs.noise_rv.get_stats()

        # scale matrix variables
        scale = (dof - 2)/dof
        self.x_smat_fi = scale * self.x_cov_fi  # turn initial covariance into initial scale matrix
        self.q_smat = scale * self.q_cov
        self.r_smat = scale * self.r_cov
        self.x_smat_pr, self.y_smat_pr, self.xy_smat = None, None, None

        self.dof = dof
        self.fixed_dof = fixed_dof

        super(StudentianInference, self).__init__(mod_dyn, mod_obs, tf_dyn, tf_obs)

    def reset(self):
        """Reset internal variables and flags."""
        self.x_mean_fi, self.x_cov_fi, self.dof_fi = self.x0_mean, self.x0_cov, self.x0_dof
        scale = (self.dof - 2) / self.dof
        self.x_smat_fi = scale * self.x_cov_fi
        self.x_smat_pr, self.y_smat_pr, self.xy_smat = None, None, None
        super(StudentianInference, self).reset()

    def _time_update(self, time, theta_dyn=None, theta_obs=None):
        """
        Time update for Studentian filters and smoothers, computing predictive moments of state and measurement.

        Parameters
        ----------
        time : int
            Time step. Important for t-variant systems.

        theta_dyn : ndarray
            Parameters of the moment transform computing the predictive state moments.

        theta_obs : ndarray
            Parameters of the moment transform computing the predictive measurement moments.
        """

        if self.fixed_dof:  # fixed-DOF version

            # pick the smallest DOF
            dof_pr = np.min((self.dof_fi, self.q_dof, self.r_dof))

            # rescale filtered scale matrix?
            scale = (dof_pr - 2) / dof_pr
            # self.x_smat_fi = self.x_smat_fi * scale * self.dof_fi / (self.dof_fi - 2)

        else:  # increasing DOF version
            scale = (self.dof - 2) / self.dof

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_fi if self.mod_dyn.noise_additive else np.hstack((self.x_mean_fi, self.q_mean))
        smat = self.x_smat_fi if self.mod_dyn.noise_additive else block_diag(self.x_smat_fi, self.q_smat)
        assert mean.ndim == 1 and smat.ndim == 2

        # predicted state statistics
        # TODO: make the moment transforms take covariance matrix (instead of scale)
        self.x_mean_pr, self.x_cov_pr, self.xx_cov = self.tf_dyn.apply(self.mod_dyn.dyn_eval, mean, smat,
                                                                       np.atleast_1d(time), theta_dyn)
        # predicted covariance -> predicted scale matrix
        self.x_smat_pr = scale * self.x_cov_pr

        if self.mod_dyn.noise_additive:
            self.x_cov_pr += self.q_gain.dot(self.q_cov).dot(self.q_gain.T)
            self.x_smat_pr += self.q_gain.dot(self.q_smat).dot(self.q_gain.T)

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_pr if self.mod_obs.noise_additive else np.hstack((self.x_mean_pr, self.r_mean))
        smat = self.x_smat_pr if self.mod_obs.noise_additive else block_diag(self.x_smat_pr, self.r_smat)
        assert mean.ndim == 1 and smat.ndim == 2

        # predicted measurement statistics
        self.y_mean_pr, self.y_cov_pr, self.xy_cov = self.tf_obs.apply(self.mod_obs.meas_eval, mean, smat,
                                                                       np.atleast_1d(time), theta_obs)
        # turn covariance to scale matrix
        self.y_smat_pr = scale * self.y_cov_pr
        self.xy_smat = scale * self.xy_cov

        # in additive case, noise covariances need to be added
        if self.mod_obs.noise_additive:
            self.y_cov_pr += self.r_cov
            self.y_smat_pr += self.r_smat

        # in non-additive case, cross-covariances must be trimmed (has no effect in additive case)
        self.xy_cov = self.xy_cov[:, :self.mod_dyn.dim_in]
        self.xx_cov = self.xx_cov[:, :self.mod_dyn.dim_in]
        self.xy_smat = self.xy_smat[:, :self.mod_dyn.dim_in]

    def _measurement_update(self, y, time=None):
        """
        Measurement update for Studentian filters, which takes predictive state and measurement moments and produces
        filtered state mean and covariance.

        Parameters
        ----------
        y : (dim, ) ndarray
            Measurement vector.

        time : int
            Time step. Important for t-variant systems.

        Notes
        -----
        Implements general Studentian filter measurement update.
        """

        # scale the covariance matrices
        # scale = (self.dof - 2) / self.dof
        # self.y_cov_pr *= scale
        # self.xy_cov *= scale

        # Kalman update
        gain = cho_solve(cho_factor(self.y_smat_pr), self.xy_smat).T
        self.x_mean_fi = self.x_mean_pr + gain.dot(y - self.y_mean_pr)
        # FIXME: this isn't covariance (shouldn't be saved in x_cov_fi)
        self.x_cov_fi = self.x_smat_pr - gain.dot(self.y_smat_pr).dot(gain.T)

        # filtered covariance to filtered scale matrix
        # delta = cho_solve(cho_factor(self.y_smat_pr), y - self.y_mean_pr)
        delta = la.solve(la.cholesky(self.y_smat_pr), y - self.y_mean_pr)
        scale = (self.dof + delta.T.dot(delta)) / (self.dof + self.mod_obs.dim_out)
        self.x_smat_fi = scale * self.x_cov_fi

        # update degrees of freedom
        self.dof_fi += self.mod_obs.dim_out

    def _smoothing_update(self):
        # Student smoother has not been developed yet.
        pass


class FullySymmetricStudent(StudentianInference):
    """
    Student filter using the fully-symmetric moment transforms from [1]_.

    Parameters
    ----------
    degree : int, optional
        Degree of the fully-symmetric quadrature rule. Degrees 3 and 5 implemented.

    kappa : float, optional
        Scaling parameter of the sigma-points of the quadrature rule.

    dof : float, optional
        Degrees of freedom of that the Student filter will maintain (on each measurement update) if `fixed_dof=True`.

    fixed_dof : bool, optional
        If `True` the filter will maintain degrees of freedom on a fixed value given by `dof`. This option preserves
        the heavy-tailed behavior. If `False`, the degrees of freedom of the filtered posterior will increase with each
        measurement update and hence the asymptotic behavior of the Student filter will be identical to that of the
        Kalman filter (the heavy-tailed behaviour is lost).

    References
    ----------
    .. [1] Tronarp, F. and Hostettler, R. and Särkkä, S. Sigma-point Filtering for Nonlinear Systems with Non-additive
           Heavy-tailed Noise, 19th International Conference on Information Fusion, 2016, 1859-1866
    """

    def __init__(self, dyn, obs, degree=3, kappa=None, dof=4.0, fixed_dof=True):
        dyn_dof = np.min((dyn.init_rv.dof, dyn.noise_rv.dof))
        obs_dof = np.min((dyn_dof, obs.noise_rv.dof))
        t_dyn = FullySymmetricStudentTransform(dyn.dim_in, degree, kappa, dyn_dof)
        t_obs = FullySymmetricStudentTransform(obs.dim_in, degree, kappa, obs_dof)
        super(FullySymmetricStudent, self).__init__(dyn, obs, t_dyn, t_obs, dof, fixed_dof)


class StudentProcessStudent(StudentianInference):
    """
    Student's t-process quadrature Student filter (TPQSF, see [1]_) with fully-symmetric sigma-points (see [3]_).

    Parameters
    ----------
    kern_par_dyn : ndarray
        Kernel parameters for TPQ transformation of the state moments.

    kern_par_obs : ndarray
        Kernel parameters for TPQ transformation of the measurement moments.

    point_par : dict, optional
        Hyper-parameters of the sigma-point set.

    dof : float, optional
        Desired degrees of freedom during the filtering process.

    dof_tp : float, optional
        Degrees of freedom of the Student's t-regression model.

    fixed_dof : bool, optional
        Fix degrees of freedom during filtering. If `True`, preserves the heavy-tailed behavior of the Student
        filter with increasing time steps. If `False`, the Student filter measurement update rule effectively becomes
        identical to the Kalman filter with increasing number of processed measurements.

    References
    ----------
    .. [1] Prüher, J.; Tronarp, F.; Karvonen, T.; Särkkä, S. and Straka, O. Student-t Process Quadratures for
           Filtering of Non-linear Systems with Heavy-tailed Noise, 20th International Conference on Information
           Fusion, 2017 , 1-8

    .. [2] Tronarp, F. and Hostettler, R. and Särkkä, S. Sigma-point Filtering for Nonlinear Systems with Non-additive
           Heavy-tailed Noise, 19th International Conference on Information Fusion, 2016, 1859-1866

    .. [3] J. McNamee and F. Stenger, Construction of fully symmetric numerical integration formulas,
           Numerische Mathematik, vol. 10, pp. 327–344, 1967
    """

    def __init__(self, dyn, obs, kern_par_dyn, kern_par_obs, point_par=None, dof=4.0, fixed_dof=True, dof_tp=4.0):
        # degrees of freedom for SSM noises
        assert isinstance(dyn.init_rv, StudentRV) and isinstance(dyn.noise_rv, StudentRV)
        q_dof, r_dof = dyn.noise_rv.dof, obs.noise_rv.dof

        # add DOF of the noises to the sigma-point parameters
        if point_par is None:
            point_par = dict()
        point_par_dyn = point_par.copy()
        point_par_obs = point_par.copy()
        point_par_dyn.update({'dof': q_dof})
        point_par_obs.update({'dof': r_dof})
        # TODO: why is q_dof parameter for unit-points of the dynamics?

        t_dyn = StudentTProcessTransform(dyn.dim_in, 1, kern_par_dyn, 'rbf-student', 'fs', point_par_dyn, nu=dof_tp)
        t_obs = StudentTProcessTransform(obs.dim_in, 1, kern_par_obs, 'rbf-student', 'fs', point_par_obs, nu=dof_tp)
        super(StudentProcessStudent, self).__init__(dyn, obs, t_dyn, t_obs, dof, fixed_dof)


"""
Warning: EXPERIMENTAL!

Inference algorithms using 'truncated' transforms that account for the fact that the measurement models do not have to 
use the whole state to compute measurements.
"""


class TruncatedUnscentedKalman(GaussianInference):
    """
    Truncated cubature Kalman filter and smoother, aware of the effective dimension of the observation model.

    Parameters
    ----------
    dyn : TransitionModel
        Transition model defining the system dynamics with Gaussian noise and initial conditions.

    obs : MeasurementModel
        Measurement model with Gaussian noise.
    """

    def __init__(self, dyn, obs, kappa=None, alpha=1.0, beta=2.0):
        tf = UnscentedTransform(dyn.dim_in, kappa, alpha, beta)
        th = TruncatedUnscentedTransform(obs.dim_state, obs.dim_in, kappa, alpha, beta)
        super(TruncatedUnscentedKalman, self).__init__(dyn, obs, tf, th)


class TruncatedCubatureKalman(GaussianInference):
    """
    Truncated cubature Kalman filter and smoother, aware of the effective dimension of the observation model.

    Parameters
    ----------
    dyn : TransitionModel
        Transition model defining the system dynamics with Gaussian noise and initial conditions.

    obs : MeasurementModel
        Measurement model with Gaussian noise.
    """

    def __init__(self, dyn, obs):
        tf = SphericalRadialTransform(dyn.dim_in)
        th = TruncatedSphericalRadialTransform(obs.dim_state, obs.dim_in)
        super(TruncatedCubatureKalman, self).__init__(dyn, obs, tf, th)


class TruncatedGaussHermiteKalman(GaussianInference):
    """
    Truncated Gauss-Hermite Kalman filter and smoother, aware of the effective dimensionality of the observation model.

    Parameters
    ----------
    dyn : TransitionModel
        Transition model defining the system dynamics with Gaussian noise and initial conditions.

    obs : MeasurementModel
        Measurement model with Gaussian noise.

    degree : int, optional
        Degree of the Gauss-Hermite integration rule. Determines the number of sigma-points.
    """

    def __init__(self, dyn, obs, degree):
        tf = GaussHermiteTransform(dyn.dim_in, degree)
        th = TruncatedGaussHermiteTransform(obs.dim_state, dyn.dim_in, degree)
        super(TruncatedGaussHermiteKalman, self).__init__(dyn, obs, tf, th)


"""
Warning: EXPERIMENTAL!

Inference algorithms based on Bayesian quadrature with multi-output GP/TP models.
"""


class MultiOutputGaussianProcessKalman(GaussianInference):
    """
    Gaussian process quadrature Kalman filter and smoother with multi-output Gaussian process model.

    Parameters
    ----------
    kern_par_dyn : (dim_out, num_par) ndarray
        Kernel parameters for GPQ transformation of the state moments. One row per output.

    kern_par_obs : (dim_out, num_par) ndarray
        Kernel parameters for GPQ transformation of the measurement moments. One row per output.

    kernel : str {'rbf'}, optional
        Kernel (covariance function) of the internal Gaussian process regression model.

    points : str {'sr', 'ut', 'gh', 'fs'}, optional
        Sigma-point set:

        ``sr``
            Spherical-radial sigma-points (originally used in CKF).
        ``ut``
            Unscented transform sigma-points (originally used in UKF).
        ``gh``
            Gauss-Hermite sigma-points (originally used in GHKF).
        ``fs``
            Fully-symmetric sigma-points [3]_ (originally used in [2]_).

    point_par : dict, optional
        Hyper-parameters of the sigma-point set.

    References
    ----------
    .. [1] Prüher, J. and Straka, O. Gaussian Process Quadrature Moment Transform,
           IEEE Transactions on Automatic Control, 2017, Pre-print, 1-1

    .. [2] Tronarp, F. and Hostettler, R. and Särkkä, S. Sigma-point Filtering for Nonlinear Systems with Non-additive
           Heavy-tailed Noise, 19th International Conference on Information Fusion, 2016, 1859-1866

    .. [3] J. McNamee and F. Stenger, Construction of fully symmetric numerical integration formulas,
           Numerische Mathematik, vol. 10, pp. 327–344, 1967

    Notes
    -----
    For experimental purposes only. Frequently breaks down with loss of positive definiteness!

    """

    def __init__(self, dyn, obs, kern_par_dyn, kern_par_obs, kernel='rbf', points='ut', point_hyp=None):
        t_dyn = MultiOutputGaussianProcessTransform(dyn.dim_in, dyn.dim_state, kern_par_dyn, kernel, points, point_hyp)
        t_obs = MultiOutputGaussianProcessTransform(obs.dim_in, obs.dim_out, kern_par_obs, kernel, points, point_hyp)
        super(MultiOutputGaussianProcessKalman, self).__init__(dyn, obs, t_dyn, t_obs)


class MultiOutputStudentProcessStudent(StudentianInference):
    """
    Student's t-process quadrature Student filter (TPQSF, see [1]_) with fully-symmetric sigma-points (see [3]_) and
    multi-output Student's t-process regression model.

    Parameters
    ----------
    kern_par_dyn : ndarray
        Kernel parameters for TPQ transformation of the state moments.

    kern_par_obs : ndarray
        Kernel parameters for TPQ transformation of the measurement moments.

    point_par : dict, optional
        Hyper-parameters of the sigma-point set.

    dof : float, optional
        Desired degrees of freedom during the filtering process.

    dof_tp : float, optional
        Degrees of freedom of the Student's t-regression model.

    fixed_dof : bool, optional
        Fix degrees of freedom during filtering. If `True`, preserves the heavy-tailed behavior of the Student
        filter with increasing time steps. If `False`, the Student filter measurement update rule effectively becomes
        identical to the Kalman filter with increasing number of processed measurements.

    Notes
    -----
    Just experimental, it doesn't work! Frequently fails with loss of positive definiteness.

    References
    ----------
    .. [1] Prüher, J.; Tronarp, F.; Karvonen, T.; Särkkä, S. and Straka, O. Student-t Process Quadratures for
           Filtering of Non-linear Systems with Heavy-tailed Noise, 20th International Conference on Information
           Fusion, 2017 , 1-8

    .. [2] Tronarp, F. and Hostettler, R. and Särkkä, S. Sigma-point Filtering for Nonlinear Systems with Non-additive
           Heavy-tailed Noise, 19th International Conference on Information Fusion, 2016, 1859-1866

    .. [3] J. McNamee and F. Stenger, Construction of fully symmetric numerical integration formulas,
           Numerische Mathematik, vol. 10, pp. 327–344, 1967
    """

    def __init__(self, dyn, obs, kern_par_dyn, kern_par_obs, point_par=None, dof=4.0, fixed_dof=True, dof_tp=4.0):
        # degrees of freedom for SSM noises
        q_dof, r_dof = dyn.get_pars('q_dof', 'r_dof')

        # add DOF of the noises to the sigma-point parameters
        if point_par is None:
            point_par = dict()
        point_par_dyn = point_par.copy()
        point_par_obs = point_par.copy()
        point_par_dyn.update({'dof': q_dof})
        point_par_obs.update({'dof': r_dof})

        t_dyn = MultiOutputStudentTProcessTransform(dyn.dim_in, dyn.dim_state, kern_par_dyn,
                                                    'rbf-student', 'fs', point_par_dyn, nu=dof_tp)
        t_obs = MultiOutputStudentTProcessTransform(obs.dim_in, obs.dim_out, kern_par_obs,
                                                    'rbf-student', 'fs', point_par_obs, nu=dof_tp)
        super(MultiOutputStudentProcessStudent, self).__init__(dyn, obs, t_dyn, t_obs, dof, fixed_dof)


"""
Warning: EXPERIMENTAL!

Inference algorithms that marginalize moment transform parameters.
"""


class MarginalInference(GaussianInference):
    """
    Gaussian state-space inference with marginalized moment transform parameters. Standard Gaussian is used as a
    prior on log-parameters (the parameters are assumed strictly positive) of the moment transforms.

    Parameters
    ----------
    par_mean : (num_par, ) ndarray
        Mean of the Gaussian prior over moment transform parameters.

    par_cov : (num_par, num_par) ndarray
        Covariance of the Gaussian prior over moment transform parameters.

    Notes
    -----
    Aims to be used mainly with Bayesian quadrature transforms, although, in principle, any moment transform with
    parameters fits into this framework.

    Warning
    -------
    Purely for experimental purposes!
    """

    def __init__(self, dyn, obs, tf_dyn, tf_obs, par_mean=None, par_cov=None):
        super(MarginalInference, self).__init__(dyn, obs, tf_dyn, tf_obs)

        # Number of parameters for each moment transform and total number of parameters
        # TODO: Generalize, transform should provide number of parameters (sum of kernel, model and point parameters)
        self.param_dyn_dim = self.mod_dyn.dim_in + 1
        self.param_obs_dim = self.mod_obs.dim_state + 1
        self.param_dim = self.param_dyn_dim + self.param_obs_dim

        # Log-parameter prior mean and covariance
        self.param_prior_mean = np.zeros(self.param_dim, ) if par_mean is None else par_mean
        self.param_prior_cov = np.eye(self.param_dim) if par_cov is None else par_cov
        # Log-parameter posterior moments initialized with prior
        self.param_mean = self.param_prior_mean
        self.param_cov = self.param_prior_cov
        # Jitter for parameter vector
        self.param_jitter = 1e-8 * np.eye(self.param_dim)

        # Spherical-radial quadrature rule for marginalizing transform parameters
        from ssmtoybox.mtran import SphericalRadialTransform
        self.param_upts = SphericalRadialTransform.unit_sigma_points(self.param_dim)
        self.param_wts = SphericalRadialTransform.weights(self.param_dim)
        self.param_pts_num = self.param_upts.shape[1]

    def reset(self):
        """Reset internal variables and flags."""
        super(MarginalInference, self).reset()
        # Reset parameter moments to prior moments
        self.param_mean = self.param_prior_mean
        self.param_cov = self.param_prior_cov

    def _measurement_update(self, y, time=None):
        """
        Computes the posterior state mean and covariance by marginalizing out the moment transform parameters.

        Procedure has two steps:
          1. Compute Laplace approximation of the GPQ parameter posterior
          2. Use fully-symmetric quadrature rule to compute posterior state mean and covariance by marginalizing
             out the GPQ-parameters over the approximated posterior.

        Parameters
        ----------
        y: (dim, ) ndarray
            Measurement vector.
        time : int
            Time step. Important for t-variant systems.
        """

        # Mean and covariance of the parameter posterior by Laplace approximation
        self._param_posterior_moments(y, time)

        # Marginalization of moment transform parameters
        param_cov_chol = la.cholesky(self.param_cov)
        param_pts = self.param_mean[:, na] + param_cov_chol.dot(self.param_upts)
        mean = np.zeros((self.mod_dyn.dim_in, self.param_pts_num))
        cov = np.zeros((self.mod_dyn.dim_in, self.mod_dyn.dim_in, self.param_pts_num))

        # Evaluate state posterior with different values of transform parameters
        for i in range(self.param_pts_num):
            # FIXME: fcn recomputes predictive estimates (x_mean_pr, x_cov_pr, ...)
            # FIXME: predictive moments should be computed by quadrature, based on param prior
            mean[:, i], cov[:, :, i] = self._state_posterior_moments(param_pts[:, i], y, time)

        # Weighted sum of means and covariances approximates Gaussian mixture state posterior
        self.x_mean_fi = np.einsum('ij, j -> i', mean, self.param_wts)
        self.x_cov_fi = np.einsum('ijk, k -> ij', cov, self.param_wts)

    def _state_posterior_moments(self, theta, y, k):
        """
        State posterior moments given moment transform parameters :math:`N(x_k | y_1:k, theta)`.

        Parameters
        ----------
        theta : ndarray
            Moment transform parameters.

        y : ndarray
            Observations.

        k : int
            Time index.

        Returns
        -------
        mean : ndarray
            Conditional state posterior mean.

        cov : ndarray
            Conditional state posterior covariance.
        """

        # Dynamics and observation model parameters
        theta_dyn, theta_obs = np.exp(theta[:self.param_dyn_dim]), np.exp(theta[self.param_dyn_dim:])

        # Moments of the joint N(x_k, y_k | y_1:k-1, theta)
        self._time_update(k, theta_dyn, theta_obs)

        # Moments of the conditional state posterior N(x_k | y_1:k, theta)
        gain = cho_solve(cho_factor(self.y_cov_pr), self.xy_cov).T
        mean = self.x_mean_pr + gain.dot(y - self.y_mean_pr)
        cov = self.x_cov_pr - gain.dot(self.y_cov_pr).dot(gain.T)
        return mean, cov

    def _param_log_likelihood(self, theta, y, k):
        """
        :math:`l(theta) = N(y_k | m_k^y(theta), P_k^y(theta))`

        Parameters
        ----------
        theta : ndarray
            Vector of transform parameters.

        y : ndarray
            Observation.

        k : int
            Time (for time varying dynamics).

        Returns
        -------
            Value of likelihood for given vector of parameters and observation.
        """

        # Dynamics and observation model parameters, convert from log-space
        theta_dyn, theta_obs = np.exp(theta[:self.param_dyn_dim]), np.exp(theta[self.param_dyn_dim:])

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_fi if self.mod_dyn.noise_additive else np.hstack((self.x_mean_fi, self.q_mean))
        cov = self.x_cov_fi if self.mod_dyn.noise_additive else block_diag(self.x_cov_fi, self.q_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute predicted state mean, covariance
        mean, cov, ccov = self.tf_dyn.apply(self.mod_dyn.dyn_eval, mean, cov, np.atleast_1d(k), theta_dyn)
        if self.mod_dyn.noise_additive:
            cov += self.G.dot(self.q_cov).dot(self.G.T)

        # in non-additive case, augment mean and covariance
        mean = mean if self.mod_obs.noise_additive else np.hstack((mean, self.r_mean))
        cov = cov if self.mod_obs.noise_additive else block_diag(cov, self.r_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute measurement mean, covariance
        mean, cov, ccov = self.tf_obs.apply(self.mod_obs.meas_eval, mean, cov, np.atleast_1d(k), theta_obs)
        if self.mod_obs.noise_additive:
            cov += self.r_cov

        return multivariate_normal.logpdf(y, mean, cov)

    def _param_log_prior(self, theta):
        """
        Prior on transform parameters :math:`p(\\theta) = N(\\theta | m^{\\theta}_k-1, P^{\\theta}_k-1)`.

        Parameters
        ----------
        theta : ndarray
            Vector of transform parameters.

        Returns
        -------
        log_prob : (ndarray or scalar)
            Log of the probability density function evaluated at theta

        Notes
        -----
        At the moment, only Gaussian prior is supported. Student-t prior might be implemented in the future.
        """
        return multivariate_normal.logpdf(theta, self.param_mean, self.param_cov)

    def _param_neg_log_posterior(self, theta, y, k):
        """
        Un-normalized negative log-posterior over transform parameters.

        Parameters
        ----------
        theta : ndarray
            Moment transform parameters.

        y : ndarray
            Observation.

        k : int
            Time index.

        Returns
        -------
        x: float
            Evaluation of un-normalized negative log-posterior over transform parameters.
        """
        return -self._param_log_likelihood(theta, y, k) - self._param_log_prior(theta)

    def _param_posterior_moments(self, y, k):
        """
        Laplace approximation of the intractable transform parameter posterior.

        Parameters
        ----------
        y : ndarray
            Observation.

        k : int
            Time index.
        """

        from scipy.optimize import minimize
        # Find theta_* = arg_max log N(y_k | m(theta), P(theta)) + log N(theta | mu, Pi)

        # Initial guess; PROBLEM: which initial guess to choose?
        # random from prior
        # theta_0 = np.random.multivariate_normal(self.param_prior_mean, self.param_prior_cov, 1)
        # theta_0 = self.param_prior_mean
        # random from previous posterior
        # theta_0 = np.random.multivariate_normal(self.param_mean, self.param_cov, 1).squeeze()
        theta_0 = self.param_mean
        # Solver options

        opt_res = minimize(self._param_neg_log_posterior, theta_0, (y, k), method='BFGS')
        self.param_mean, self.param_cov = opt_res.x, opt_res.hess_inv + self.param_jitter


class MarginalizedGaussianProcessKalman(MarginalInference):
    """
    Gaussian process quadrature Kalman filter and smoother with marginalized GPQ moment transform kernel parameters.

    Notes
    -----
    For experimental purposes only. Likely a dead-end!
    """

    def __init__(self, dyn, obs, kernel='rbf', points='ut', point_hyp=None, par_mean=None, par_cov=None):
        # arbitrary dummy kernel parameters, because transforms wouldn't initialize
        kpar_dyn = np.ones((1, dyn.dim_in + 1))
        kpar_obs = np.ones((1, obs.dim_state + 1))

        t_dyn = GaussianProcessTransform(dyn.dim_in, 1, kpar_dyn, kernel, points, point_hyp)
        t_obs = GaussianProcessTransform(obs.dim_state, 1, kpar_obs, kernel, points, point_hyp)
        super(MarginalizedGaussianProcessKalman, self).__init__(dyn, obs, t_dyn, t_obs, par_mean, par_cov)


"""
Warning: EXPERIMENTAL!

Extended Kalman filter via Gaussian process quadrature with derivative evaluations.
"""


class ExtendedKalmanGPQD(GaussianInference):
    """
    Extended Kalman filter and smoother with a moment transform based on single-point Gaussian process quadrature with
    derivative observations and RBF kernel.

    Parameters
    ----------
    rbf_par_dyn : (1, dim_in+1) ndarray
        RBF kernel parameters for the dynamics.

    rbf_par_obs : (1, dim_in+1) ndarray
        RBF kernel parameters for the measurement model.
    """

    def __init__(self, dyn, obs, rbf_par_dyn, rbf_par_obs):
        tf = TaylorGPQDTransform(dyn.dim_in, rbf_par_dyn)
        th = TaylorGPQDTransform(obs.dim_state, rbf_par_obs)
        super(ExtendedKalmanGPQD, self).__init__(dyn, obs, tf, th)
