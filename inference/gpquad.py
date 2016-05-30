import numpy as np

from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import GPQuad
from transforms.quad import Unscented


class GPQuadKalman(StateSpaceInference):
    """
    GP quadrature filter and smoother.
    """

    def __init__(self, sys, usp_dyn=None, usp_meas=None, hyp_dyn=None, hyp_meas=None):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        self.tf = GPQuad(nq, usp_dyn, hyp_dyn)
        self.th = GPQuad(nr, usp_meas, hyp_meas)
        super(GPQuadKalman, self).__init__(self.tf, self.th, sys)


def main():
    # UNGM demo
    # from models.ungm import ungm_filter_demo
    # ungm_filter_demo(GPQuadKalman, hyp_dyn=hdyn, hyp_meas=hmeas)

    # Pendulum demo
    # from models.pendulum import pendulum_filter_demo
    # hdyn, hmeas = None, None
    # pendulum_filter_demo(GPQuadKalman, hyp_dyn=hdyn, hyp_meas=hmeas)

    # Reentry vehicle tracking demo
    # from models.tracking import bot_filter_demo, reentry_filter_demo
    # d = 5
    # hdyn = {'sig_var': 1.0, 'lengthscale': 25.0 * np.ones(d, ), 'noise_var': 1e-8}
    # hmeas = {'sig_var': 1.0, 'lengthscale': 25.0 * np.ones(d, ), 'noise_var': 1e-8}
    # usp = Unscented.unit_sigma_points(d, kappa=0.0)  # kappa=3-d, alpha=1.0
    # # usp = None
    # reentry_filter_demo(GPQuadKalman, usp_dyn=usp, usp_meas=usp, hyp_dyn=hdyn, hyp_meas=hmeas)

    # Frequency demodulation demo
    d = 2
    hdyn = {'sig_var': 10.0, 'lengthscale': 30.0 * np.ones(d, ), 'noise_var': 1e-8}
    hmeas = {'sig_var': 10.0, 'lengthscale': 30.0 * np.ones(d, ), 'noise_var': 1e-8}
    usp = Unscented.unit_sigma_points(d, kappa=0.0)  # kappa=3-d, alpha=1.0
    from models.demodulation import frequency_demodulation_filter_demo
    frequency_demodulation_filter_demo(GPQuadKalman, usp_dyn=usp, usp_meas=usp, hyp_dyn=hdyn, hyp_meas=hmeas)


if __name__ == '__main__':
    main()
