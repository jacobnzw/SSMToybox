from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.linalg as la
from scipy.linalg import cho_factor, cho_solve, block_diag
from scipy.stats import multivariate_normal
from numpy import newaxis as na
from models.ssmodel import StateSpaceModel, GaussianStateSpaceModel, StudentStateSpaceModel
from transforms.mtform import MomentTransform


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
        self.x_mean_fi, self.x_cov_fi = self.ssm.get_pars('x0_mean', 'x0_cov')
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


class MarginalInference(StateSpaceInference):
    """
    Kalman state-space inference with marginalized moment transform parameters. Standard Gaussian is used as a
    prior on log-parameters (the parameters are assumed strictly positive). Aims to be used mainly with Bayesian
    quadrature transforms, although, in principle, any moment transform with parameters fits into this framework.


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
        from transforms.quad import SphericalRadial
        self.param_upts = SphericalRadial.unit_sigma_points(self.param_dim)
        self.param_wts = SphericalRadial.weights(self.param_dim)
        self.param_pts_num = self.param_upts.shape[1]

    def reset(self):
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
        y: ndarray
          Measurement at a given time step

        Returns
        -------

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
        State posterior moments given moment transform parameters N(x_k | y_1:k, theta).

        Parameters
        ----------
        theta: ndarray
            Moment transform parameters.
        y: ndarray
            Observations.
        k: int
            Time index.

        Returns
        -------
            (mean, cov): (ndarray, ndarray)
                Conditional state posterior mean and covariance.
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
        l(theta) = N(y_k | m_k^y(theta), P_k^y(theta))

        Parameters
        ----------
        theta: ndarray
            Vector of transform parameters.
        y: ndarray
            Observation
        k: int
            Time (for time varying dynamics)

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
        Prior on transform parameters.

        p(theta) = N(theta | m^theta_k-1, P^theta_k-1)

        Parameters
        ----------
        theta: ndarray
            Vector of transform parameters.

        Notes
        -----
        At the moment, only Gaussian prior is supported. Student-t prior might be implemented in the future.

        Returns
        -------
        p(theta): return type of scipy.stats.multivariate_normal.pdf
            Value of a Gaussian prior PDF.

        """
        return multivariate_normal.logpdf(theta, self.param_mean, self.param_cov)

    def _param_neg_log_posterior(self, theta, y, k):
        """
        Un-normalized negative log-posterior over transform parameters.

        Parameters
        ----------
        theta: ndarray
            Transform parameters
        y: ndarray
            Observation
        k: int
            Time

        Returns
        -------
        x: float
            Evaluation of un-normalized negative logarithm of posterior over transform parameters.
        """
        return -self._param_log_likelihood(theta, y, k) - self._param_log_prior(theta)

    def _param_posterior_moments(self, y, k):
        """
        Laplace approximation of the intractable transform parameter posterior.

        Parameters
        ----------
        y: ndarray
            Observation
        k: int
            Time

        Returns
        -------
        (mean, cov): tuple
            Mean and covariance of the intractable parameter posterior.
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
