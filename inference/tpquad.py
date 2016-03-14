from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import TPQuad


class TPQuadKalman(StateSpaceInference):
    """
    T-Process-quadrature filter and smoother.
    """

    def __init__(self, sys, usp_dyn, usp_meas, hyp_dyn, hyp_meas):
        assert isinstance(sys, StateSpaceModel)
        self.tf = TPQuad(usp_dyn, hyp_dyn, nu=2.5)
        self.th = TPQuad(usp_meas, hyp_meas, nu=2.5)
        super(TPQuadKalman, self).__init__(self.tf, self.th, sys)


def main():
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(TPQuadKalman)


if __name__ == '__main__':
    main()
