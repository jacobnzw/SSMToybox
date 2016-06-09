from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.taylor import Taylor1stOrder, TaylorGPQD


class ExtendedKalman(StateSpaceInference):
    """
    Extended Kalman filter/smoother. For linear system functions this is a Kalman filter.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Taylor1stOrder(nq)
        th = Taylor1stOrder(nr)
        super(ExtendedKalman, self).__init__(sys, tf, th)


class ExtendedKalmanGPQD(StateSpaceInference):
    def __init__(self, sys, alpha=1.0, el=1.0):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = TaylorGPQD(nq, alpha, el)
        th = TaylorGPQD(nr, alpha, el)
        super(ExtendedKalmanGPQD, self).__init__(sys, tf, th)


def main():
    # from models.ungm import ungm_filter_demo
    # ungm_filter_demo(ExtendedKalman)
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(ExtendedKalmanGPQD, alpha=1.0, ell=1.0)
    # from models.pendulum import pendulum_filter_demo
    # pendulum_filter_demo(ExtendedKalman)


if __name__ == '__main__':
    main()
