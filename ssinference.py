import numpy as np
from scipy.linalg import cho_factor, cho_solve
from transform import Unscented


class StateSpaceInference(object):

    def __init__(self, trans, sys):
        self.trans = trans  # transformation decides which filter will be used
        self.sys = sys  # system whose state is to be estimated
        self.x_mean_filt, self.x_cov_filt, self.q_cov, self.r_cov = sys.get_pars(
                'x0_mean', 'x0_cov', 'q_cov', 'r_cov'
        )

        self.flags = {'filt_state_available': False}

    def get_flag(self, key):
        return self.flags[key]

    def set_flag(self, key, value):
        self.flags[key] = value

    def forward_pass(self, data):
        D, N = data.shape
        self.filt_state_mean = np.empty((D, N))
        self.filt_state_cov = np.empty((D, D, N))
        self.pred_state_mean = self.filt_state_mean.copy()
        self.pred_state_cov = self.filt_state_cov.copy()
        for k in xrange(data.shape[-1]):  # iterate over columns of data
            self._time_update(k)
            self.pred_state_mean[:, k], self.pred_state_cov[:, :, k] = self.x_mean_pred, self.x_cov_pred
            self._measurement_update(data[:, k])
            self.filt_state_mean[:, k], self.filt_state_cov[:, :, k] = self.x_mean_filt, self.x_cov_filt
        # set flag that filtered state sequence is available
        self.set_flag('filt_state_available', True)
        return self.filt_state_mean, self.filt_state_cov

    def backward_pass(self):
        assert self.get_flag('filt_state_available')  # filtered state is required for now, condition in the future?
        D, N = self.filt_state_mean.shape
        state_mean_smoothed = np.empty((D, N))
        state_cov_smoothed = np.empty((D, D, N))
        for k in xrange(N-1, 0, -1):
            self._smoothing_update()
            state_mean_smoothed[:, k] = self.x_mean_smooth
            state_cov_smoothed[:, :, k] = self.x_cov_smooth
        return state_mean_smoothed, state_cov_smoothed

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
    system = UNGM(10, 1)
    time_steps = 200
    X, Z = system.simulate(time_steps, 1)  # get some data from the system
    infer = UKFS(system)
    mean_f, cov_f = infer.forward_pass(Z[..., 0])

    # plot trajectory with measurements
    import matplotlib.pyplot as plt
    # plt.plot(X[0, ...], color='b', alpha=0.15)
    # plt.plot(Z[0, ...], color='k', alpha=0.25, ls='None', marker='.')
    plt.plot(X[0, :, 0], color='r', ls='--')
    plt.plot(Z[0, :, 0], color='k', ls='None', marker='o')
    plt.plot(mean_f[0, ...], color='b')
    plt.fill_between(range(0, time_steps),
                     mean_f[0, ...] - 2*np.sqrt(cov_f[0, 0, :]),
                     mean_f[0, ...] + 2*np.sqrt(cov_f[0, 0, :]),
                     color='b', alpha=0.15)

if __name__ == '__main__':
    main()
