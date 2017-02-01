from inference.ssinfer import StateSpaceInference, GaussianInference
from models.ssmodel import StateSpaceModel
from transforms.quad import GaussHermite, GaussHermiteTrunc


class GaussHermiteKalman(GaussianInference):
    """
    Gauss-Hermite Kalman filter and smoother.
    """

    def __init__(self, sys, deg=3):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = GaussHermite(nq, degree=deg)
        th = GaussHermite(nr, degree=deg)
        super(GaussHermiteKalman, self).__init__(sys, tf, th)


class GaussHermiteTruncKalman(GaussianInference):
    """
    Truncated Gauss-Hermite Kalman filter and smoother. Aware of the effective dimensionality.
    """

    def __init__(self, sys, deg=3):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = GaussHermite(nq, degree=deg)
        th = GaussHermiteTrunc(nr, sys.rD, degree=deg)
        super(GaussHermiteTruncKalman, self).__init__(sys, tf, th)


def main():
    # from models.ungm import ungm_filter_demo
    # ungm_filter_demo(GaussHermiteKalman, deg=10)
    from models.pendulum import pendulum_filter_demo
    pendulum_filter_demo(GaussHermiteKalman, deg=10)


if __name__ == '__main__':
    main()
