from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.linalg as la
from scipy.linalg import cho_factor, cho_solve, block_diag
from scipy.stats import multivariate_normal
from numpy import newaxis as na
from ssmtoybox.ssmod import StateSpaceModel, StudentStateSpaceModel
from ssmtoybox.bq.bqmtran import GaussianProcessTransform, GPQMO, StudentTProcessTransform, TPQMO, BayesSardTransform
from ssmtoybox.mtran import MomentTransform, Taylor1stOrder, TaylorGPQD, SphericalRadial, SphericalRadialTrunc, \
    Unscented, UnscentedTrunc, GaussHermite, GaussHermiteTrunc


class StateSpaceInference(metaclass=ABCMeta):
    """
    Base class for all local state-space inference algorithms, including nonlinear filters and smoothers.

    Parameters
    ----------
    ssm : StateSpaceModel
        State-space model to perform inference on.

    tf_dyn : MomentTransform
        Moment transform for computing predictive state moments.

    tf_meas : MomentTransform
        Moment transform for computing predictive measurement moments.

    """

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

    Parameters
    ----------
    ssm : GaussianStateSpaceModel
        State-space model to perform inference on.

    tf_dyn : MomentTransform
        Moment transform for computing predictive state moments.

    tf_meas : MomentTransform
        Moment transform for computing predictive measurement moments.
    """

    def __init__(self, ssm, tf_dyn, tf_meas):

        # dynamical system whose state is to be estimated
        assert isinstance(ssm, StateSpaceModel)

        # set initial condition mean and covariance, and noise covariances
        self.x_mean_fi, self.x_cov_fi, self.q_mean, self.q_cov, self.r_mean, self.r_cov, self.G = ssm.get_pars(
            'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov', 'q_gain'
        )

        super(GaussianInference, self).__init__(ssm, tf_dyn, tf_meas)

    def reset(self):
        """Reset internal variables and flags."""
        self.x_mean_fi, self.x_cov_fi = self.ssm.get_pars('x0_mean', 'x0_cov')
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


class StudentInference(StateSpaceInference):
    """
    Base class for state-space inference algorithms, which assume that the state and measurement variables are jointly
    Student's t-distributed.

    Parameters
    ----------
    ssm : StudentStateSpaceModel
        Studentian State space model to perform inference on. Must implement the 'q_dof' and 'r_dof' properties.

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

    Notes
    -----
    Even though Student's t distribution is not parametrized by the covariance matrix like the Gaussian,
    the filter still produces mean and covariance of the state.

    """

    def __init__(self, ssm, tf_dyn, tf_meas, dof=4.0, fixed_dof=True):
        # require Student state-space model
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
        """Reset internal variables and flags."""
        self.x_mean_fi, self.x_cov_fi, self.dof_fi = self.ssm.get_pars('x0_mean', 'x0_cov', 'x0_dof')
        scale = (self.dof - 2) / self.dof
        self.x_smat_fi = scale * self.x_cov_fi
        self.x_smat_pr, self.y_smat_pr, self.xy_smat = None, None, None
        super(StudentInference, self).reset()

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


class MarginalInference(GaussianInference):
    """
    Gaussian state-space inference with marginalized moment transform parameters. Standard Gaussian is used as a
    prior on log-parameters (the parameters are assumed strictly positive) of the moment transforms.

    Parameters
    ----------
    ssm : GaussianStateSpaceModel
        State-space model to perform inference on.

    tf_dyn : MomentTransform
        Moment transform for computing predictive state moments.

    tf_meas : MomentTransform
        Moment transform for computing predictive measurement moments.

    par_mean : ndarray
        Mean of the Gaussian prior over moment transform parameters.

    par_cov : ndarray
        Covariance of the Gaussian prior over moment transform parameters.

    Notes
    -----
    Aims to be used mainly with Bayesian quadrature transforms, although, in principle, any moment transform with
    parameters fits into this framework.

    Warning
    -------
    Purely for experimental purposes!
    """

    def __init__(self, ssm, tf_dyn, tf_meas, par_mean=None, par_cov=None):
        super(MarginalInference, self).__init__(ssm, tf_dyn, tf_meas)

        # Number of parameters for each moment transform and total number of parameters
        # TODO: Generalize, transform should provide number of parameters (sum of kernel, model and point parameters)
        self.param_dyn_dim = self.ssm.xD + 1
        self.param_obs_dim = self.ssm.xD + 1
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
        from mtran import SphericalRadial
        self.param_upts = SphericalRadial.unit_sigma_points(self.param_dim)
        self.param_wts = SphericalRadial.weights(self.param_dim)
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
        mean = np.zeros((self.ssm.xD, self.param_pts_num))
        cov = np.zeros((self.ssm.xD, self.ssm.xD, self.param_pts_num))

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
        mean = self.x_mean_fi if self.ssm.q_additive else np.hstack((self.x_mean_fi, self.q_mean))
        cov = self.x_cov_fi if self.ssm.q_additive else block_diag(self.x_cov_fi, self.q_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute predicted state mean, covariance
        mean, cov, ccov = self.tf_dyn.apply(self.ssm.dyn_eval, mean, cov, self.ssm.par_fcn(k), theta_dyn)
        if self.ssm.q_additive:
            cov += self.G.dot(self.q_cov).dot(self.G.T)

        # in non-additive case, augment mean and covariance
        mean = mean if self.ssm.r_additive else np.hstack((mean, self.r_mean))
        cov = cov if self.ssm.r_additive else block_diag(cov, self.r_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute measurement mean, covariance
        mean, cov, ccov = self.tf_meas.apply(self.ssm.meas_eval, mean, cov, self.ssm.par_fcn(k), theta_obs)
        if self.ssm.r_additive:
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


class ExtendedKalman(GaussianInference):
    """
    Extended Kalman filter and smoother. For linear system functions this is a Kalman filter/smoother.

    Parameters
    ----------
    sys : GaussianStateSpaceModel
        State-space model to perform inference on. Needs to have Jacobians implemented.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Taylor1stOrder(nq)
        th = Taylor1stOrder(nr)
        super(ExtendedKalman, self).__init__(sys, tf, th)


class ExtendedKalmanGPQD(GaussianInference):
    """
    Extended Kalman filter and smoother with a moment transform based on single-point Gaussian process quadrature with
    derivative observations and RBF kernel.

    Parameters
    ----------
    sys : GaussianStateSpaceModel
        State-space model to perform inference on. Needs to have Jacobians implemented.

    alpha : float, optional
        Scaling parameter of the RBF kernel.

    el : float, optional
        Input scaling parameter (a.k.a. horizontal length-scale) of the RBF kernel.
    """
    def __init__(self, sys, alpha=1.0, el=1.0):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = TaylorGPQD(nq, alpha, el)
        th = TaylorGPQD(nr, alpha, el)
        super(ExtendedKalmanGPQD, self).__init__(sys, tf, th)


class CubatureKalman(GaussianInference):
    """
    Cubature Kalman filter and smoother.

    Parameters
    ----------
    sys : GaussianStateSpaceModel
        State-space model to perform inference on.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = SphericalRadial(nq)
        th = SphericalRadial(nr)
        super(CubatureKalman, self).__init__(sys, tf, th)


class CubatureTruncKalman(GaussianInference):
    """
    Truncated cubature Kalman filter and smoother, aware of the effective dimension of the observation model.

    Parameters
    ----------
    sys : GaussianStateSpaceModel
        State-space model to perform inference on.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = SphericalRadial(nq)
        th = SphericalRadialTrunc(nr, sys.rD)
        super(CubatureTruncKalman, self).__init__(sys, tf, th)


class UnscentedKalman(GaussianInference):
    """
    Unscented Kalman filter and smoother.

    Parameters
    ----------
    sys : GaussianStateSpaceModel
        State-space model to perform inference on.
    """

    def __init__(self, sys, kappa=None, alpha=1.0, beta=2.0):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Unscented(nq, kappa=kappa, alpha=alpha, beta=beta)
        th = Unscented(nr, kappa=kappa, alpha=alpha, beta=beta)
        super(UnscentedKalman, self).__init__(sys, tf, th)


class UnscentedTruncKalman(GaussianInference):
    """
    Truncated cubature Kalman filter and smoother, aware of the effective dimension of the observation model.

    Parameters
    ----------
    sys : GaussianStateSpaceModel
        State-space model to perform inference on.
    """

    def __init__(self, sys, kappa=None, alpha=1.0, beta=2.0):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Unscented(nq, kappa=kappa, alpha=alpha, beta=beta)
        th = UnscentedTrunc(nr, sys.rD, kappa=kappa, alpha=alpha, beta=beta)
        super(UnscentedTruncKalman, self).__init__(sys, tf, th)


class GaussHermiteKalman(GaussianInference):
    """
    Gauss-Hermite Kalman filter and smoother.

    Parameters
    ----------
    sys : GaussianStateSpaceModel
        State-space model to perform inference on.

    deg : int, optional
        Degree of the Gauss-Hermite integration rule. Determines the number of sigma-points.
    """

    def __init__(self, sys, deg=3):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = GaussHermite(nq, degree=deg)
        th = GaussHermite(nr, degree=deg)
        super(GaussHermiteKalman, self).__init__(sys, tf, th)


class GaussHermiteTruncKalman(GaussianInference):
    """
    Truncated Gauss-Hermite Kalman filter and smoother, aware of the effective dimensionality of the observation model.

    Parameters
    ----------
    sys : GaussianStateSpaceModel
        State-space model to perform inference on.

    deg : int, optional
        Degree of the Gauss-Hermite integration rule. Determines the number of sigma-points.
    """

    def __init__(self, sys, deg=3):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = GaussHermite(nq, degree=deg)
        th = GaussHermiteTrunc(nr, sys.rD, degree=deg)
        super(GaussHermiteTruncKalman, self).__init__(sys, tf, th)


class GaussianProcessKalman(GaussianInference):
    """
    Gaussian process quadrature Kalman filter (GPQKF) and smoother (see [1]_).

    Parameters
    ----------
    ssm : GaussianStateSpaceModel
        State-space model to perform inference on.

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

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, kernel='rbf', points='ut', point_hyp=None):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = GaussianProcessTransform(nq, kern_par_dyn, kernel, points, point_hyp)
        t_obs = GaussianProcessTransform(nr, kern_par_obs, kernel, points, point_hyp)
        super(GaussianProcessKalman, self).__init__(ssm, t_dyn, t_obs)


class BayesSardKalman(GaussianInference):
    """
    Bayes-Sard quadrature Kalman filter (BSQKF) and smoother.

    Parameters
    ----------
    ssm : GaussianStateSpaceModel
        State-space model to perform inference on.

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
    """
    def __init__(self, ssm, kern_par_dyn, kern_par_obs, mulind_dyn=2, mulind_obs=2, points='ut', point_hyp=None):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = BayesSardTransform(nq, kern_par_dyn, mulind_dyn, points, point_hyp)
        t_obs = BayesSardTransform(nr, kern_par_obs, mulind_obs, points, point_hyp)
        super(BayesSardKalman, self).__init__(ssm, t_dyn, t_obs)


class GPQMOKalman(GaussianInference):
    """
    Gaussian process quadrature Kalman filter and smoother with multi-output Gaussian process model.

    Parameters
    ----------
    ssm : GaussianStateSpaceModel
        State-space model to perform inference on.

    ker_par_dyn : ndarray
        Kernel parameters for GPQ transformation of the state moments.

    ker_par_obs : ndarray
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

    def __init__(self, ssm, ker_par_dyn, ker_par_obs, kernel='rbf', points='ut', point_par=None):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = GPQMO(nq, ssm.xD, ker_par_dyn, kernel, points, point_par)
        t_obs = GPQMO(nr, ssm.zD, ker_par_obs, kernel, points, point_par)
        super(GPQMOKalman, self).__init__(ssm, t_dyn, t_obs)


class GPQMKalman(MarginalInference):
    """
    Gaussian process quadrature Kalman filter and smoother with marginalized GPQ moment transform kernel parameters.

    Notes
    -----
    For experimental purposes only. Likely a dead-end!
    """
    def __init__(self, sys, kernel, points, par_mean=None, par_cov=None, point_hyp=None):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        t_dyn = GaussianProcessTransform(nq, kernel, points, point_par=point_hyp)
        t_obs = GaussianProcessTransform(nr, kernel, points, point_par=point_hyp)
        super(GPQMKalman, self).__init__(sys, t_dyn, t_obs, par_mean, par_cov)


class TPQKalman(GaussianInference):
    """
    Student's t-process quadrature Kalman filter (TPQKF) and smoother (see [1]_).

    Parameters
    ----------
    ssm : GaussianStateSpaceModel
        Gaussian state-space model to perform inference on.

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

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, kernel='rbf', points='ut', point_hyp=None, nu=3.0):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = StudentTProcessTransform(nq, kern_par_dyn, kernel, points, point_hyp, nu)
        t_obs = StudentTProcessTransform(nr, kern_par_obs, kernel, points, point_hyp, nu)
        super(TPQKalman, self).__init__(ssm, t_dyn, t_obs)


class TPQStudent(StudentInference):
    """
    Student's t-process quadrature Student filter (TPQSF, see [1]_) with fully-symmetric sigma-points (see [3]_).

    Parameters
    ----------
    ssm : StudentStateSpaceModel
        Studentian state-space model to perform inference on.

    kern_par_dyn : ndarray
        Kernel parameters for TPQ transformation of the state moments.

    kern_par_obs : ndarray
        Kernel parameters for TPQ transformation of the measurement moments.

    point_par : dict, optional
        Hyper-parameters of the sigma-point set.

    dof : float, optional
        Desired degrees of freedom during the filtering process.

    dof_tp : float, optional
        Degrees of freedom of the Student's t-regression model. TODO: could be merged with `fixed_dof`.

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

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, point_par=None, dof=4.0, fixed_dof=True, dof_tp=4.0):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD

        # degrees of freedom for SSM noises
        q_dof, r_dof = ssm.get_pars('q_dof', 'r_dof')

        # add DOF of the noises to the sigma-point parameters
        if point_par is None:
            point_par = dict()
        point_par_dyn = point_par
        point_par_obs = point_par
        point_par_dyn.update({'dof': q_dof})
        point_par_obs.update({'dof': r_dof})
        # TODO: finish fixing DOFs, DOF for TPQ and DOF for the filtered state.

        t_dyn = StudentTProcessTransform(nq, kern_par_dyn, 'rbf-student', 'fs', point_par_dyn, nu=dof_tp)
        t_obs = StudentTProcessTransform(nr, kern_par_obs, 'rbf-student', 'fs', point_par_obs, nu=dof_tp)
        super(TPQStudent, self).__init__(ssm, t_dyn, t_obs, dof, fixed_dof)


class TPQMOStudent(StudentInference):
    """
    Student's t-process quadrature Student filter (TPQSF, see [1]_) with fully-symmetric sigma-points (see [3]_) and
    multi-output Student's t-process regression model.

    Parameters
    ----------
    ssm : StudentStateSpaceModel
        Studentian state-space model to perform inference on.

    ker_par_dyn : ndarray
        Kernel parameters for TPQ transformation of the state moments.

    ker_par_obs : ndarray
        Kernel parameters for TPQ transformation of the measurement moments.

    point_par : dict, optional
        Hyper-parameters of the sigma-point set.

    dof : float, optional
        Desired degrees of freedom during the filtering process.

    dof_tp : float, optional
        Degrees of freedom of the Student's t-regression model. TODO: could be merged with `fixed_dof`.

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

    def __init__(self, ssm, ker_par_dyn, ker_par_obs, point_par=None, dof=4.0, fixed_dof=True, dof_tp=4.0):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD

        # degrees of freedom for SSM noises
        q_dof, r_dof = ssm.get_pars('q_dof', 'r_dof')

        # add DOF of the noises to the sigma-point parameters
        if point_par is None:
            point_par = dict()
        point_par_dyn = point_par
        point_par_obs = point_par
        point_par_dyn.update({'dof': q_dof})
        point_par_obs.update({'dof': r_dof})

        t_dyn = TPQMO(nq, ssm.xD, ker_par_dyn, 'rbf-student', 'fs', point_par_dyn, nu=dof_tp)
        t_obs = TPQMO(nr, ssm.zD, ker_par_obs, 'rbf-student', 'fs', point_par_obs, nu=dof_tp)
        super(TPQMOStudent, self).__init__(ssm, t_dyn, t_obs, dof, fixed_dof)