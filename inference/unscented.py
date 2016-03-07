from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.quad import Unscented


class UnscentedKalman(StateSpaceInference):
    """
    Unscented Kalman filter and smoother.
    """

    def __init__(self, sys, kap=None, al=1.0, bet=2.0):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Unscented(nq, kappa=kap, alpha=al, beta=bet)
        th = Unscented(nr, kappa=kap, alpha=al, beta=bet)
        super(UnscentedKalman, self).__init__(tf, th, sys)


def main():
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(UnscentedKalman, kap=0.0)


if __name__ == '__main__':
    main()
