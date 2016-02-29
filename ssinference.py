import numpy as np
from scipy.linalg import cho_factor, cho_solve
from transform import Unscented


class StateSpaceInference(object):

    def __init__(self, trans, sys):
        self.trans = trans  # transformation decides which filter will be used
        self.sys = sys  # dynamical system whose state is to be estimated
        # set initial condition mean and covariance, and noise covariances
        self.x_mean_filt, self.x_cov_filt, self.q_cov, self.r_cov = sys.get_pars(
                'x0_mean', 'x0_cov', 'q_cov', 'r_cov'
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
        self.fi_mean = np.empty((self.D, self.N))
        self.fi_cov = np.empty((self.D, self.D, self.N))
        self.pr_mean = self.fi_mean.copy()
        self.pr_cov = self.fi_cov.copy()
        self.pr_xx_cov = self.fi_cov.copy()
        for k in xrange(self.N):  # iterate over columns of data
            self._time_update(k)
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
        for k in xrange(self.N-2, 0, -1):
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

    def _time_update(self, time):
        # calculate predictive moments of the system state by applying moment transformation
        self.x_mean_pred, self.x_cov_pred, self.xx_cov = self.trans.apply(
                self.sys.dyn_fcn, self.x_mean_filt, self.x_cov_filt, self.sys.par_fcn(time)
        )
        self.mean_z_pred, self.z_cov_pred, self.xz_cov = self.trans.apply(
                self.sys.meas_fcn, self.x_mean_pred, self.x_cov_pred, self.sys.par_fcn(time)
        )
        # TODO: following holds only in additive noise case, rethink how to handle this in general
        self.x_cov_pred += self.q_cov
        self.z_cov_pred += self.r_cov

    def _measurement_update(self, y):
        gain = cho_solve(cho_factor(self.z_cov_pred), self.xz_cov)
        self.x_mean_filt = self.x_mean_pred + gain.dot(y - self.mean_z_pred)
        self.x_cov_filt = self.x_cov_pred - gain.dot(self.z_cov_pred).dot(gain.T)

    def _smoothing_update(self):
        gain = cho_solve(cho_factor(self.x_cov_pred), self.xx_cov)
        self.x_mean_smooth = self.x_mean_filt + gain.dot(self.x_mean_smooth - self.x_mean_pred)
        self.x_cov_smooth = self.x_cov_filt + gain.dot(self.x_cov_smooth - self.x_cov_pred).dot(gain.T)


class UKFS(StateSpaceInference):
    """
    Unscented Kalman filter and smoother.
    """

    def __init__(self, sys):
        t = Unscented(sys.Dx, kappa=0)  # UKFS has-a Unscented transform
        super(UKFS, self).__init__(t, sys)


def main():
    from ssm import UNGM
    system = UNGM(q_cov=10, r_cov=.1)
    time_steps = 50
    X, Z = system.simulate(time_steps, 1)  # get some data from the system
    # plot trajectory with measurements
    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(X[0, ...], color='b', alpha=0.15)
    # plt.plot(Z[0, ...], color='k', alpha=0.25, ls='None', marker='.')

    infer = UKFS(system)
    mean_f, cov_f = infer.forward_pass(Z[..., 0])
    mean_s, cov_s = infer.backward_pass()

    plt.figure()
    plt.plot(X[0, :, 0], color='r', ls='--')
    plt.plot(Z[0, :, 0], color='k', ls='None', marker='o')
    plt.plot(mean_f[0, ...], color='b')
    plt.fill_between(range(0, time_steps),
                     mean_f[0, ...] - 2*np.sqrt(cov_f[0, 0, :]),
                     mean_f[0, ...] + 2*np.sqrt(cov_f[0, 0, :]),
                     color='b', alpha=0.15)
    plt.plot(mean_s[0, ...], color='g')
    plt.fill_between(range(0, time_steps),
                     mean_s[0, ...] - 2*np.sqrt(cov_s[0, 0, :]),
                     mean_s[0, ...] + 2*np.sqrt(cov_s[0, 0, :]),
                     color='g', alpha=0.25)

if __name__ == '__main__':
    main()
