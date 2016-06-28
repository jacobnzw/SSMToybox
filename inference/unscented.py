from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.quad import Unscented, UnscentedTrunc


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
        super(UnscentedKalman, self).__init__(sys, tf, th)


class UnscentedTruncKalman(StateSpaceInference):
    """
    Unscented Kalman filter and smoother.
    """

    def __init__(self, sys, kappa=None, alpha=1.0, beta=2.0):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Unscented(nq, kappa=kappa, alpha=alpha, beta=beta)
        th = UnscentedTrunc(nr, sys.rD, kappa=kappa, alpha=alpha, beta=beta)
        super(UnscentedTruncKalman, self).__init__(sys, tf, th)


def main():
    # from models.ungm import ungm_filter_demo
    # ungm_filter_demo(UnscentedKalman, kappa=0.0)
    # from models.pendulum import pendulum_filter_demo
    # pendulum_filter_demo(UnscentedKalman)
    from models.tracking import bot_filter_demo, reentry_filter_demo
    reentry_filter_demo(UnscentedTruncKalman)
    # from models.demodulation import frequency_demodulation_filter_demo
    # frequency_demodulation_filter_demo(UnscentedKalman)

if __name__ == '__main__':
    main()


# Running UnscentedKalman filter/smoother (750 time steps, 100 MC simulations) ...
# Filter stats:
# =============
# Time-averaged RMSE (position): 0.0117935920012
# Time-averaged RMSE (velocity): 0.0357979024925
# Smoother stats:
# ===============
# Time-averaged RMSE (position): 0.00644206658631
# Time-averaged RMSE (velocity): 0.0168587712893
