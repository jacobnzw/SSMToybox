import numpy as np
from scipy.linalg import cho_factor, cho_solve, block_diag

from models.ssmodel import StateSpaceModel
from transforms.mtform import MomentTransform


class StateSpaceInference(object):
    def __init__(self, ssm, transf_dyn, transf_meas):
        # separate moment transforms for system dynamics and measurement model
        assert isinstance(transf_dyn, MomentTransform) and isinstance(transf_meas, MomentTransform)
        self.transf_dyn = transf_dyn
        self.transf_meas = transf_meas
        # dynamical system whose state is to be estimated
        assert isinstance(ssm, StateSpaceModel)
        self.ssm = ssm
        # set initial condition mean and covariance, and noise covariances
        self.x_mean_filt, self.x_cov_filt, self.q_mean, self.q_cov, self.r_mean, self.r_cov, self.G = ssm.get_pars(
            'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov', 'q_factor'
        )
        self.flags = {'filtered': False, 'smoothed': False}
        self.x_mean_pred, self.x_cov_pred, = None, None
        self.x_mean_smooth, self.x_cov_smooth = None, None
        self.xx_cov, self.xz_cov = None, None
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
        self.fi_mean = np.zeros((self.ssm.xD, self.N))
        self.fi_cov = np.zeros((self.ssm.xD, self.ssm.xD, self.N))
        self.fi_mean[:, 0], self.fi_cov[..., 0] = self.x_mean_filt, self.x_cov_filt
        self.pr_mean = self.fi_mean.copy()
        self.pr_cov = self.fi_cov.copy()
        self.pr_xx_cov = self.fi_cov.copy()
        for k in range(1, self.N):  # iterate over columns of data
            self._time_update(k - 1)
            self.pr_mean[..., k] = self.x_mean_pred
            self.pr_cov[..., k] = self.x_cov_pred
            self.pr_xx_cov[..., k] = self.xx_cov
            self._measurement_update(data[:, k])
            self.fi_mean[..., k], self.fi_cov[..., k] = self.x_mean_filt, self.x_cov_filt
        # set flag that filtered state sequence is available
        self.set_flag('filtered', True)
        # smoothing estimate at the last time step == the filtering estimate at the last time step
        self.x_mean_smooth, self.x_cov_smooth = self.x_mean_filt, self.x_cov_filt
        return self.fi_mean, self.fi_cov

    def backward_pass(self):
        assert self.get_flag('filtered')  # require filtered state
        self.sm_mean = self.fi_mean.copy()
        self.sm_cov = self.fi_cov.copy()
        for k in range(self.N-2, 0, -1):
            self.x_mean_pred = self.pr_mean[..., k+1]
            self.x_cov_pred = self.pr_cov[..., k+1]
            self.xx_cov = self.pr_xx_cov[..., k+1]
            self.x_mean_filt = self.fi_mean[..., k]
            self.x_cov_filt = self.fi_cov[..., k]
            self._smoothing_update()
            self.sm_mean[..., k] = self.x_mean_smooth
            self.sm_cov[..., k] = self.x_cov_smooth
        self.set_flag('smoothed', True)
        return self.sm_mean, self.sm_cov

    def reset(self):
        self.x_mean_filt, self.x_cov_filt = self.ssm.get_pars('x0_mean', 'x0_cov')
        self.flags = {'filtered': False, 'smoothed': False}
        self.x_mean_pred, self.x_cov_pred, = None, None
        self.x_mean_smooth, self.x_cov_smooth = None, None
        self.xx_cov, self.xz_cov = None, None
        self.pr_mean, self.pr_cov, self.pr_xx_cov = None, None, None
        self.fi_mean, self.fi_cov = None, None
        self.sm_mean, self.sm_cov = None, None
        self.D, self.N = None, None

    def _time_update(self, time):
        # in non-additive case, augment mean and covariance
        mean = self.x_mean_filt if self.ssm.q_additive else np.hstack((self.x_mean_filt, self.q_mean))
        cov = self.x_cov_filt if self.ssm.q_additive else block_diag(self.x_cov_filt, self.q_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute predicted state mean, covariance
        self.x_mean_pred, self.x_cov_pred, self.xx_cov = self.transf_dyn.apply(self.ssm.dyn_eval, mean, cov,
                                                                               self.ssm.par_fcn(time))
        if self.ssm.q_additive:
            self.x_cov_pred += self.G.dot(self.q_cov).dot(self.G.T)

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_pred if self.ssm.r_additive else np.hstack((self.x_mean_pred, self.r_mean))
        cov = self.x_cov_pred if self.ssm.r_additive else block_diag(self.x_cov_pred, self.r_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute measurement mean, covariance
        self.z_mean_pred, self.z_cov_pred, self.xz_cov = self.transf_meas.apply(self.ssm.meas_eval, mean, cov,
                                                                                self.ssm.par_fcn(time))
        # in additive case, noise covariances need to be added
        if self.ssm.r_additive:
            self.z_cov_pred += self.r_cov

        # in non-additive case, cross-covariances must be trimmed (has no effect in additive case)
        self.xz_cov = self.xz_cov[:, :self.ssm.xD]
        self.xx_cov = self.xx_cov[:, :self.ssm.xD]

    def _measurement_update(self, y):
        gain = cho_solve(cho_factor(self.z_cov_pred), self.xz_cov).T
        self.x_mean_filt = self.x_mean_pred + gain.dot(y - self.z_mean_pred)
        self.x_cov_filt = self.x_cov_pred - gain.dot(self.z_cov_pred).dot(gain.T)

    def _smoothing_update(self):
        gain = cho_solve(cho_factor(self.x_cov_pred), self.xx_cov).T
        self.x_mean_smooth = self.x_mean_filt + gain.dot(self.x_mean_smooth - self.x_mean_pred)
        self.x_cov_smooth = self.x_cov_filt + gain.dot(self.x_cov_smooth - self.x_cov_pred).dot(gain.T)


class MarginalInference(StateSpaceInference):

    def _time_update(self, time):
        # in non-additive case, augment mean and covariance
        mean = self.x_mean_filt if self.ssm.q_additive else np.hstack((self.x_mean_filt, self.q_mean))
        cov = self.x_cov_filt if self.ssm.q_additive else block_diag(self.x_cov_filt, self.q_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute predicted state mean, covariance
        self.x_mean_pred, self.x_cov_pred, self.xx_cov = self.transf_dyn.apply(self.ssm.dyn_eval, mean, cov,
                                                                               self.ssm.par_fcn(time))
        if self.ssm.q_additive:
            self.x_cov_pred += self.G.dot(self.q_cov).dot(self.G.T)

        # in non-additive case, augment mean and covariance
        mean = self.x_mean_pred if self.ssm.r_additive else np.hstack((self.x_mean_pred, self.r_mean))
        cov = self.x_cov_pred if self.ssm.r_additive else block_diag(self.x_cov_pred, self.r_cov)
        assert mean.ndim == 1 and cov.ndim == 2

        # apply moment transform to compute measurement mean, covariance
        self.z_mean_pred, self.z_cov_pred, self.xz_cov = self.transf_meas.apply(self.ssm.meas_eval, mean, cov,
                                                                                self.ssm.par_fcn(time))
        # in additive case, noise covariances need to be added
        if self.ssm.r_additive:
            self.z_cov_pred += self.r_cov

        # in non-additive case, cross-covariances must be trimmed (has no effect in additive case)
        self.xz_cov = self.xz_cov[:, :self.ssm.xD]
        self.xx_cov = self.xx_cov[:, :self.ssm.xD]

    def _measurement_update(self, y):
        """
        Computes the posterior state mean and covariance by marginalizing out the GPQ-parameters.

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

        # Laplace approximation:
        #   Input:
        #       - function handle for J(theta) = log l(theta) + log q(theta | y_1:k-1)
        #         where l(theta) = p(y_k | theta) = N(y_k | m_k^y(theta), P_k^y(theta))
        #   Output:
        #       - mean and covariance of parameter posterior q(theta | y_1:k)

        # Marginalization of GPQ-parameters
        #   Input:
        #       - moments of q(theta | y_1:k)
        #       - function for evaluating posterior mean and covariance at theta

        gain = cho_solve(cho_factor(self.z_cov_pred), self.xz_cov).T
        self.x_mean_filt = self.x_mean_pred + gain.dot(y - self.z_mean_pred)
        self.x_cov_filt = self.x_cov_pred - gain.dot(self.z_cov_pred).dot(gain.T)

    def _smoothing_update(self):
        gain = cho_solve(cho_factor(self.x_cov_pred), self.xx_cov).T
        self.x_mean_smooth = self.x_mean_filt + gain.dot(self.x_mean_smooth - self.x_mean_pred)
        self.x_cov_smooth = self.x_cov_filt + gain.dot(self.x_cov_smooth - self.x_cov_pred).dot(gain.T)

    def _param_likelihood(self, theta, y, k):
        """
        l(theta) = p(y_k | theta) = N(y_k | m_k^y(theta), P_k^y(theta))

        Parameters
        ----------
        theta: ndarray
            Vector of quadrature parameters.
        y: ndarray
            Measurement y_k
        k: ndarray
            time (for time varying dynamics)

        Returns
        -------
            Value of likelihood for given vector of parameters and observation.
        """
        # TODO: compute mean and covariance to evaluate N(y_k | m_k^y(theta), P_k^y(theta))
        pass

    def _param_prior(self, theta):
        """
        Prior on quadrature parameters. So far only Gaussian on log-parameters.

        p(theta) = N(theta | m^theta_k-1, P^theta_k-1)

        Parameters
        ----------
        theta: ndarray
            Vector of quadrature parameters.

        Returns
        -------
            Value of a Gaussian prior PDF.

        """
        pass

    def _param_posterior_moments(self):
        # Laplace approximation of p(theta | y_k) is a Gaussian
        # 1. find theta_* = arg_max p(theta | y_k) ==> mean
        # 2. evaluate Hessian of p(theta | y_k) at theta_* ==> covariance
        pass
