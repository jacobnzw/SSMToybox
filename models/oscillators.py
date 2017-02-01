from inference.ssinfer import StateSpaceInference
from models.ssmodel import GaussianStateSpaceModel
import matplotlib.pyplot as plt
import numpy as np


class VanDerPol(GaussianStateSpaceModel):

    xD = 2  # state dimension
    zD = 1  # measurement dimension
    qD = 2  # state noise dimension
    rD = 1  # measurement noise dimension

    q_additive = True  # True = state noise is additive, False = non-additive
    r_additive = True

    def __init__(self, dt=0.1, alpha=1):
        self.dt = dt
        self.alpha = alpha
        req_kwargs = {
            'x0_mean': np.zeros(self.xD),
            'x0_cov': np.diag([6.3e-4, 2.2e-4]),
            'q_mean': np.zeros(self.qD),
            'q_cov': np.diag([2.62e-2, 8e-3]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[3e-3]]),
            'q_gain': np.eye(2)
        }
        super(VanDerPol, self).__init__(**req_kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] + self.dt*x[1], x[1] + self.dt*(self.alpha*x[1]*(1 - x[0]**2) - x[0])])

    def meas_fcn(self, x, r, pars):
        return np.array([x[1]]) + r

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def par_fcn(self, time):
        pass


def vanderpol_demo():
    steps = 250
    mc_simulations = 10
    ssm = VanDerPol()
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)

    plt.figure()
    plt.subplot(211)
    plt.plot(x[0, ...], color='b', lw=2, alpha=0.25, label='x_1[k]')
    plt.subplot(212)
    plt.plot(x[1, ...], color='b', lw=2, alpha=0.25, label='x_1[k]')

    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')

    # plt.figure()
    # plt.plot(x[0, :, 0], x[1, :, 0])
    plt.show()


if __name__ == '__main__':
    vanderpol_demo()
