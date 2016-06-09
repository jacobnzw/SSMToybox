import numpy as np

from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import TPQ


class TPQKalman(StateSpaceInference):
    """
    T-Process-quadrature filter and smoother.
    """

    def __init__(self, sys, kernel, points, kern_hyp_dyn=None, kern_hyp_obs=None, point_hyp=None, nu=None):
        # nu = None gets passed all the way down to the StudenTProcess which decides default value
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        t_dyn = TPQ(nq, kernel, points, kern_hyp_dyn, point_hyp, nu)
        t_obs = TPQ(nr, kernel, points, kern_hyp_obs, point_hyp, nu)
        super(TPQKalman, self).__init__(t_dyn, t_obs, sys)


def main():
    # UNGM demo
    from models.ungm import ungm_filter_demo
    khyp = {'alpha': 1.0, 'el': 0.3 * np.ones(1)}
    ut_hyp = {'kappa': 0.0}
    ungm_filter_demo(TPQKalman, 'rbf', 'sr', kern_hyp_dyn=khyp, kern_hyp_obs=khyp, point_hyp=ut_hyp)


if __name__ == '__main__':
    main()
