from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.quad import SphericalRadial


class CubatureKalman(StateSpaceInference):
    """
    Cubature Kalman filter and smoother.
    """

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = SphericalRadial(nq)
        th = SphericalRadial(nr)
        super(CubatureKalman, self).__init__(tf, th, sys)


def main():
    from models.ungm import ungm_filter_demo
    ungm_filter_demo(CubatureKalman)


if __name__ == '__main__':
    main()
