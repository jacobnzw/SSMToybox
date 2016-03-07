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
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(GPQuadKalman)


if __name__ == '__main__':
    main()
