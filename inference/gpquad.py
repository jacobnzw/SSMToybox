import numpy as np

from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import GPQuad
from transforms.quad import Unscented


class GPQuadKalman(StateSpaceInference):
    """
    GP quadrature filter and smoother.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        unit_sp_f = Unscented.unit_sigma_points(nq, np.sqrt(nq + 0))
        unit_sp_h = Unscented.unit_sigma_points(nr, np.sqrt(nr + 0))
        hypers_f = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((nq,)), 'noise_var': 1e-8}
        hypers_h = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((nr,)), 'noise_var': 1e-8}
        self.tf = GPQuad(unit_sp_f, hypers_f)
        self.th = GPQuad(unit_sp_h, hypers_h)
        super(GPQuadKalman, self).__init__(self.tf, self.th, sys)


def main():
    from models.ungm import UNGM
    system = UNGM(q_cov=10, r_cov=1)
    print "q_additive: {}, r_additive: {}".format(system.q_additive, system.r_additive)

    time_steps, mc = 500, 50
    x, z = system.simulate(time_steps, mc_sims=mc)  # get some data from the system

    filt = GPQuadKalman(system)
    rmse_filter = np.zeros(mc)
    rmse_smoother = np.zeros(mc)
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        rmse_filter[imc] = np.sqrt(np.mean((x[:, 1:, imc] - mean_f[:, 1:]) ** 2, axis=1))
        rmse_smoother[imc] = np.sqrt(np.mean((x[:, 1:, imc] - mean_s[:, 1:]) ** 2, axis=1))
        filt.reset()
    # FIXME: results are not even close to the BHKF on the same example!
    print "Filter RMSE: {:.4f}".format(np.asscalar(rmse_filter.mean()))
    print "Smoother RMSE: {:.4f}".format(np.asscalar(rmse_smoother.mean()))

    import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    main()
