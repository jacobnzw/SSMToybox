import numpy as np

from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import GPQuadDer
from transforms.quad import Unscented


class GPQuadDerKalman(StateSpaceInference):
    """
    GP quadrature filter and smoother.
    """

    def __init__(self, sys, usp_dyn=None, usp_meas=None, hyp_dyn=None, hyp_meas=None):
        assert isinstance(sys, StateSpaceModel)
        if usp_dyn is not None and usp_meas is not None and hyp_dyn is not None and hyp_meas is not None:
            self.usp_dyn, self.usp_meas, self.hyp_dyn, self.hyp_meas = usp_dyn, usp_meas, hyp_dyn, hyp_meas
        else:
            self._set_default_usp_hyp(sys)
        self.tf = GPQuadDer(self.usp_dyn, self.hyp_dyn)
        self.th = GPQuadDer(self.usp_meas, self.hyp_meas)
        super(GPQuadDerKalman, self).__init__(self.tf, self.th, sys)

    def _set_default_usp_hyp(self, sys):
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        self.usp_dyn = Unscented.unit_sigma_points(nq, np.sqrt(nq + 0))
        self.usp_meas = Unscented.unit_sigma_points(nr, np.sqrt(nr + 0))
        self.hyp_dyn = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((nq,)), 'noise_var': 1e-8}
        self.hyp_meas = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((nr,)), 'noise_var': 1e-8}


def main():
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(GPQuadDerKalman)
    # from models.pendulum import pendulum_filter_demo
    # pendulum_filter_demo(GPQuadKalman)


if __name__ == '__main__':
    main()
