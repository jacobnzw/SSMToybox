import numpy as np

from inference.ssinfer import StateSpaceInference, StudentInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import TPQ


class TPQKalman(StateSpaceInference):
    """
    T-Process-quadrature filter and smoother for the Gaussian inference.
    """

    def __init__(self, sys, kernel, points, kern_hyp_dyn=None, kern_hyp_obs=None, point_hyp=None, nu=None):
        # nu = None gets passed all the way down to the StudenTProcess which decides default value
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        t_dyn = TPQ(nq, kernel, points, kern_hyp_dyn, point_hyp, nu)
        t_obs = TPQ(nr, kernel, points, kern_hyp_obs, point_hyp, nu)
        super(TPQKalman, self).__init__(sys, t_dyn, t_obs)


class TPQStudent(StudentInference):
    """
    T-process quadrature filter and smoother for the Student's t inference. Uses RQ kernel and fully-symmetric
    point-sets by default. RQ kernel expectations w.r.t. Student's t-density are expressed as a simplified scale
    mixture representation which facilitates analytical tractability.
    """

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, kernel='rq', points='fs', point_hyp=None, nu=3.0):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = TPQ(nq, kern_par_dyn, kernel, points, point_hyp, nu)
        t_obs = TPQ(nr, kern_par_obs, kernel, points, point_hyp, nu)
        super(TPQStudent, self).__init__(ssm, t_dyn, t_obs)


def main():
    # UNGM demo
    from models.ungm import ungm_filter_demo
    ker_par = np.array([[1, 1, 0.3]])
    pts_par = {'kappa': 0.0}
    ungm_filter_demo(TPQStudent, ker_par, ker_par, kernel='rq', points='fs', point_hyp=pts_par)


if __name__ == '__main__':
    main()
