from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.quad import GaussHermite


class GaussHermiteKalman(StateSpaceInference):
    """
    Gauss-Hermite Kalman filter and smoother.
    """

    def __init__(self, sys, deg=3):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = GaussHermite(nq, degree=deg)
        th = GaussHermite(nr, degree=deg)
        super(GaussHermiteKalman, self).__init__(tf, th, sys)


def main():
    # from models.ungm import ungm_filter_demo
    # ungm_filter_demo(GaussHermiteKalman, deg=10)
    from models.pendulum import pendulum_filter_demo
    pendulum_filter_demo(GaussHermiteKalman, deg=10)


if __name__ == '__main__':
    main()
