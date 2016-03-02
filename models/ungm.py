import numpy as np

from models.ssmodel import StateSpaceModel


class UNGM(StateSpaceModel):
    """
    Univariate Non-linear Growth Model frequently used as a benchmark.
    """

    # CLASS VARIABLES are shared by all instances of this class
    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1
    q_additive = True
    r_additive = True

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        """
        Inits the UNGM object where state covariance (q_cov) and measurement covariance (r_cov) must be supplied. The
        initial state mean and covariance, if not supplied, will default to $x_0 ~ N(0, 1)$.
        :param q_cov: state noise covariance
        :param r_cov: measurement noise covariance
        :param x0_mean: initial state mean
        :param x0_cov: initial state covariance
        :param kwargs: additional arguments
        :return:
        """
        super(UNGM, self).__init__(**kwargs)
        self.set_pars('x0_mean', np.atleast_1d(x0_mean))
        self.set_pars('x0_cov', np.atleast_2d(x0_cov))
        self.set_pars('q_mean', np.atleast_1d(q_mean))
        self.set_pars('q_cov', np.atleast_2d(q_cov))
        self.set_pars('r_mean', np.atleast_1d(r_mean))
        self.set_pars('r_cov', np.atleast_2d(r_cov))

    def dyn_fcn(self, x, q, *pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, *pars):
        return np.asarray([0.05 * x[0] ** 2]) + r

    def par_fcn(self, time):
        return time


class UNGMnonadd(StateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    # CLASS VARIABLES are shared by all instances of this class
    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1
    q_additive = False
    r_additive = False

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        """
        Inits the UNGM object where state covariance (q_cov) and measurement covariance (r_cov) must be supplied. The
        initial state mean and covariance, if not supplied, will default to $x_0 ~ N(0, 1)$.
        :param q_cov: state noise covariance
        :param r_cov: measurement noise covariance
        :param x0_mean: initial state mean
        :param x0_cov: initial state covariance
        :param kwargs: additional arguments
        :return:
        """
        super(UNGMnonadd, self).__init__(**kwargs)
        self.set_pars('x0_mean', np.atleast_1d(x0_mean))
        self.set_pars('x0_cov', np.atleast_2d(x0_cov))
        self.set_pars('q_mean', np.atleast_1d(q_mean))
        self.set_pars('q_cov', np.atleast_2d(q_cov))
        self.set_pars('r_mean', np.atleast_1d(r_mean))
        self.set_pars('r_cov', np.atleast_2d(r_cov))

    def dyn_fcn(self, x, q, *pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * q * pars[0])])

    def meas_fcn(self, x, r, *pars):
        return np.asarray([0.05 * x[0] ** 2]) + 0.5 * r * np.sin(r)

    def par_fcn(self, time):
        return time


# TODO: spice up the UNGM demos
def ungm_demo():
    steps = 100
    mc_simulations = 50
    ssm = UNGM(q_cov=10, r_cov=.1)
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)
    import matplotlib.pyplot as plt
    plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')


def ungm_nonadd_demo():
    steps = 100
    mc_simulations = 50
    ssm = UNGMnonadd(q_cov=10, r_cov=.1)
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)
    import matplotlib.pyplot as plt
    plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')


if __name__ == '__main__':
    # TODO: input args decide what demo to run
    ungm_demo()
