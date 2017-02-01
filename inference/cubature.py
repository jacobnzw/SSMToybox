from inference.ssinfer import StateSpaceInference, GaussianInference
from models.ssmodel import StateSpaceModel
from transforms.quad import SphericalRadial, SphericalRadialTrunc


class CubatureKalman(GaussianInference):
    """
    Cubature Kalman filter and smoother.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = SphericalRadial(nq)
        th = SphericalRadial(nr)
        super(CubatureKalman, self).__init__(sys, tf, th)


class CubatureTruncKalman(GaussianInference):
    """
    Truncated cubature Kalman filter and smoother. Aware of the effective dimension of the observation model.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = SphericalRadial(nq)
        th = SphericalRadialTrunc(nr, sys.rD)
        super(CubatureTruncKalman, self).__init__(sys, tf, th)


def main():
    # from models.ungm import ungm_filter_demo
    # ungm_filter_demo(CubatureKalman)
    # from models.pendulum import pendulum_filter_demo
    # pendulum_filter_demo(CubatureKalman)
    from models.tracking import reentry_filter_demo
    reentry_filter_demo(CubatureTruncKalman)


if __name__ == '__main__':
    main()

    # Running CubatureKalman filter/smoother (750 time steps, 100 MC simulations) ...
    # Filter stats:
    # =============
    # Time-averaged RMSE (position): 0.0119559175595
    # Time-averaged RMSE (velocity): 0.0357993257056
    # Smoother stats:
    # ===============
    # Time-averaged RMSE (position): 0.00647415923241
    # Time-averaged RMSE (velocity): 0.0169948757345
