import numpy as np

from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import TPQuad
from transforms.quad import Unscented


class TPQuadKalman(StateSpaceInference):
    """
    T-Process-quadrature filter and smoother.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        unit_sp_f = Unscented.unit_sigma_points(nq, np.sqrt(nq + 0))
        unit_sp_h = Unscented.unit_sigma_points(nr, np.sqrt(nr + 0))
        hypers_f = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((nq,)), 'noise_var': 1e-8}
        hypers_h = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((nr,)), 'noise_var': 1e-8}
        self.tf = TPQuad(unit_sp_f, hypers_f, nu=2.5)
        self.th = TPQuad(unit_sp_h, hypers_h, nu=2.5)
        super(TPQuadKalman, self).__init__(self.tf, self.th, sys)


def main():
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(TPQuadKalman)


if __name__ == '__main__':
    main()
