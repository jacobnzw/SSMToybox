from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.quad import Unscented


class UnscentedKalman(StateSpaceInference):
    """
    Unscented Kalman filter and smoother.
    """

    def __init__(self, sys, kappa=None, alpha=1.0, beta=2.0):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Unscented(nq, kappa=kappa, alpha=alpha, beta=beta)
        th = Unscented(nr, kappa=kappa, alpha=alpha, beta=beta)
        super(UnscentedKalman, self).__init__(tf, th, sys)


def main():
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(UnscentedKalman, kap=0.0)
    # from models.pendulum import pendulum_filter_demo
    # pendulum_filter_demo(UnscentedKalman)


if __name__ == '__main__':
    main()
