from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.taylor import Taylor1stOrder


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
        super(ExtendedKalman, self).__init__(tf, th, sys)


def main():
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(ExtendedKalman)
    # from models.pendulum import pendulum_filter_demo
    # pendulum_filter_demo(ExtendedKalman)


if __name__ == '__main__':
    main()
