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
    # from models.ungm import ungm_filter_demo
    # ungm_filter_demo(GPQuadKalman)
    from models.pendulum import pendulum_filter_demo
    pendulum_filter_demo(GPQuadKalman)


if __name__ == '__main__':
    main()
