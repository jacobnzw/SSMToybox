from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import TPQ


class TPQuadKalman(StateSpaceInference):
    """
    T-Process-quadrature filter and smoother.
    """

    def __init__(self, sys, usp_dyn=None, usp_meas=None, hyp_dyn=None, hyp_meas=None):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        self.tf = TPQ(nq, unit_sp=usp_dyn, hypers=hyp_dyn, nu=2.5)
        self.th = TPQ(nr, unit_sp=usp_meas, hypers=hyp_meas, nu=2.5)
        super(TPQuadKalman, self).__init__(self.tf, self.th, sys)


def main():
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(TPQuadKalman)


if __name__ == '__main__':
    main()
