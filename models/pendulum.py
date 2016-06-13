import matplotlib.pyplot as plt

from inference.ssinfer import StateSpaceInference
from models.ssmodel import *


class Pendulum(StateSpaceModel):
    xD = 2  # state dimension
    zD = 1  # measurement dimension
    qD = 2
    rD = 1
    q_additive = True
    r_additive = True
    g = 9.81  # gravitation constant

    def __init__(self, x0_mean=np.array([1.5, 0]), x0_cov=0.1 * np.eye(2), r_cov=np.array([[0.32]]), dt=0.01):
        self.dt = dt
        assert x0_mean.shape == (self.xD,) and x0_cov.shape == (self.xD, self.xD)
        assert r_cov.shape == (self.zD, self.zD)
        req_kwargs = {
            'x0_mean': np.atleast_1d(x0_mean),
            'x0_cov': np.atleast_2d(x0_cov),
            'q_mean': np.zeros(self.qD),
            'q_cov': 0.01 * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.atleast_2d(r_cov),
            'q_factor': np.eye(2)
        }
        super(Pendulum, self).__init__(**req_kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] + x[1] * self.dt, x[1] - self.g * self.dt * np.sin(x[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.array([np.sin(x[0])]) + r

    def par_fcn(self, time):
        pass  # pendulum model does not have time-varying parameters

    def dyn_fcn_dx(self, x, q, pars):
        return np.array([[1.0, self.dt],
                         [-self.g * self.dt * np.cos(x[0]), 1.0]])

    def meas_fcn_dx(self, x, r, pars):
        return np.array([np.cos(x[0]), 0.0])


def pendulum_demo():
    steps = 500
    mc_simulations = 100
    ssm = Pendulum()
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)

    plt.figure()
    plt.plot(x[0, ...], color='b', lw=2, alpha=0.15, label='state trajectory')
    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')

    # plt.figure()
    # plt.plot(x[0, :, 0], x[1, :, 0])
    plt.show()


def pendulum_filter_demo(filt_class, *args, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    system = Pendulum()
    # create filter object, pass in additional kwargs
    filt = filt_class(system, *args, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 500, 100
    x, z = system.simulate(time_steps, mc_sims=mc)
    print "Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc)
    rmse_filter = np.zeros((system.xD, mc))
    rmse_smoother = np.zeros((system.xD, mc))
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        rmse_filter[:, imc] = np.sqrt(np.mean((x[..., imc] - mean_f) ** 2, axis=1))
        rmse_smoother[:, imc] = np.sqrt(np.mean((x[..., imc] - mean_s) ** 2, axis=1))
        filt.reset()
    # print average filter/smoother RMSE
    print "Filter RMSE: {}".format((rmse_filter.mean(axis=1)))
    print "Smoother RMSE: {}".format((rmse_smoother.mean(axis=1)))
    # plot one realization of the system trajectory, measurements and filtered/smoothed state estimate
    plt.figure()
    time = range(1, time_steps)
    plt.plot(x[0, :, 0], color='r', ls='--', label='true state')
    plt.plot(z[0, :, 0], color='k', ls='None', marker='o')
    plt.plot(mean_f[0, ...], color='b', label='filtered estimate')
    plt.fill_between(time,
                     mean_f[0, 1:] - 2 * np.sqrt(cov_f[0, 0, 1:]),
                     mean_f[0, 1:] + 2 * np.sqrt(cov_f[0, 0, 1:]),
                     color='b', alpha=0.15)
    plt.plot(mean_s[0, ...], color='g', label='smoothed estimate')
    plt.fill_between(time,
                     mean_s[0, 1:] - 2 * np.sqrt(cov_s[0, 0, 1:]),
                     mean_s[0, 1:] + 2 * np.sqrt(cov_s[0, 0, 1:]),
                     color='g', alpha=0.25)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    pendulum_demo()
