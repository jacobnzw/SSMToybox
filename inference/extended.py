from ssinfer import StateSpaceInference, StateSpaceModel
from transforms.taylor import Taylor1stOrder


class ExtendedKalman(StateSpaceInference):
    """
    Extended Kalman filter/smoother. For linear system functions this is a Kalman filter.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        tf = Taylor1stOrder()
        th = Taylor1stOrder()
        super(ExtendedKalman, self).__init__(sys, tf, th)
