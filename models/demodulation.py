import matplotlib.pyplot as plt

from ssmodel import *
from inference.ssinfer import StateSpaceInference


class FrequencyDemodulation(StateSpaceModel):
    """
    Frequence demodulation experiment from [1]_

    The objective is to estimate the frequency message $$ x_1 = \omega $$ from noisy in-phase and quadrature
    observations.


    References
    ==========
    .. [1] Pakki, K., et al., (2011) Cubature Information Filter and its Applications, Proceedings of the ACC 2011
    """

    xD = 2
    zD = 2
    qD = 2
    rD = 2
    q_additive = True
    r_additive = True
    # system (process) parameters
    mu = 0.9
    lam = 0.99

    def __init__(self):
        kwargs = {
            'x0_mean': np.zeros(self.xD),
            'x0_cov': np.eye(self.xD),
            'q_mean': np.zeros(self.qD),
            'q_cov': np.eye(self.qD),
            'r_mean': np.zeros(self.rD),
            'r_cov': 2e-3 * np.eye(self.rD),
            'q_factor': np.eye(self.qD),
        }
        super(FrequencyDemodulation, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([self.mu * x[0], np.arctan(self.lam * x[1] + x[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.array([np.cos(x[1]), np.sin(x[1])]) + r

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def par_fcn(self, time):
        pass


def frequency_demodulation_demo():
    steps = 100
    mc_simulations = 50
    ssm = FrequencyDemodulation()
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)

    plt.figure()
    plt.plot(x[0, ...], color='b', alpha=0.15, label='state')
    plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.show()


def frequency_demodulation_filter_demo(filt_class, *args, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    system = FrequencyDemodulation()
    # create filter object, pass in additional kwargs
    filt = filt_class(system, *args, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 500, 100
    x, z = system.simulate(time_steps, mc_sims=mc)
    print "Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc)
    rmse_filter = np.zeros(mc)
    rmse_smoother = np.zeros(mc)
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        rmse_filter[imc] = np.sqrt(np.mean((x[..., imc] - mean_f) ** 2, axis=1))
        rmse_smoother[imc] = np.sqrt(np.mean((x[..., imc] - mean_s) ** 2, axis=1))
        filt.reset()
    # print average filter/smoother RMSE
    print "Filter RMSE: {:.4f}".format(np.asscalar(rmse_filter.mean()))
    print "Smoother RMSE: {:.4f}".format(np.asscalar(rmse_smoother.mean()))
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
    frequency_demodulation_demo()
