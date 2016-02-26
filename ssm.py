import numpy as np
# TODO: Abstract Base Classes to enforce the requirements of the base class on the derived classes.


class StateSpaceModel(object):

    Dx = None  # state dimension
    Dz = None  # measurement dimension

    # lists the keyword arguments currently required by the StateSpaceModel class
    _required_kwargs_ = 'x0_mean', 'x0_cov', 'q_cov', 'r_cov'

    def __init__(self, **kwargs):
        self.pars = kwargs

    def get_pars(self, *keys):
        values = []
        for k in keys:
            values.append(self.pars.get(k))
        return values

    def set_pars(self, key, value):
        self.pars[key] = value

    def dyn_fcn(self, x, q, *args):
        """
        General non-linear system dynamics, where the state noise can be non-additive. Additional system parameters are
        optional.
        :param x: system state
        :param q: state noise
        :param args: parameters of the system dynamics
        :return: system state at the next time step
        """
        raise NotImplementedError

    def meas_fcn(self, x, r, *args):
        # state measurement model
        raise NotImplementedError

    def par_fcn(self, time):
        # describes how parameter value depends on time (for time varying systems)
        raise NotImplementedError

    def simulate(self, steps, mc_sims=1):
        """
        General implementation of the SSM simulation starting from initial conditions for a given number of time steps
        :param steps: number of time steps in state trajectory
        :param mc_sims: number of trajectories to simulate (the initial state is drawn randomly)
        :return: arrays with simulated state trajectories and measurements
        """
        x0_mean, x0_cov, q_cov, r_cov = self.get_pars('x0_mean', 'x0_cov', 'q_cov', 'r_cov')
        x = np.empty((self.Dx, steps, mc_sims))
        z = np.empty((self.Dz, steps, mc_sims))
        q = np.random.multivariate_normal(np.zeros(self.Dx), q_cov, size=(mc_sims, steps)).T
        r = np.random.multivariate_normal(np.zeros(self.Dz), r_cov, size=(mc_sims, steps)).T
        x0 = np.random.multivariate_normal(x0_mean, x0_cov, size=mc_sims).T  # (D, mc_sims)
        x[:, 0, :] = x0  # store initial states at k=0
        for imc in xrange(mc_sims):
            for k in xrange(1, steps):
                theta = self.par_fcn(k)
                x[:, k, imc] = self.dyn_fcn(x[:, k-1, imc], q[:, k-1, imc], theta)
                z[:, k, imc] = self.meas_fcn(x[:, k, imc], r[:, k, imc], theta)
        return x, z


class UNGM(StateSpaceModel):
    """
    Univariate Non-linear Growth Model frequently used as a benchmark.
    """

    # CLASS VARIABLES are shared by all instances of this class
    Dx = 1  # state dimension
    Dz = 1  # measurement dimension

    def __init__(self, q_cov, r_cov, x0_mean=np.zeros((1,)), x0_cov=np.eye(1), **kwargs):
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
        self.set_pars('x0_mean', x0_mean)
        self.set_pars('x0_cov', x0_cov)
        self.set_pars('q_cov', np.atleast_2d(q_cov))
        self.set_pars('r_cov', np.atleast_2d(r_cov))

    def dyn_fcn(self, x, q, *pars):
        """
        System dynamics
        :param x: state at time k
        :param q: noise at time k
        :param pars: pars[0] = k (time)
        :return:
        """
        return np.asarray([0.5*x[0] + 25*(x[0] / (1 + x[0]**2)) + 8*np.cos(1.2*pars[0])]) + q

    def meas_fcn(self, x, r, *pars):
        return np.asarray([0.05 * x[0]**2]) + r

    def par_fcn(self, time):
        return time


def main():
    specs = {
        # 'x0_mean': np.zeros((1,)),
        # 'x0_cov': .02*np.eye(1),
        # 'q_cov': np.atleast_2d(10),
        # 'r_cov': np.atleast_2d(1)
    }
    m = UNGM(.1, .1)
    X, Z = m.simulate(100, 50)
    import matplotlib.pyplot as plt
    plt.plot(X[0, ...], color='b', alpha=0.15)
    plt.plot(Z[0, ...], color='k', alpha=0.25, ls='None', marker='.')

if __name__ == '__main__':
    main()
