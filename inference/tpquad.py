import numpy as np

from inference.ssinfer import StateSpaceInference, StudentInference, GaussianInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import TPQ


class TPQKalman(GaussianInference):
    """
    T-Process-quadrature filter and smoother for the Gaussian inference.
    """

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, kernel='rbf', points='ut', point_hyp=None, nu=3.0):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = TPQ(nq, kern_par_dyn, kernel, points, point_hyp, nu)
        t_obs = TPQ(nr, kern_par_obs, kernel, points, point_hyp, nu)
        super(TPQKalman, self).__init__(ssm, t_dyn, t_obs)


class TPQStudent(StudentInference):
    """
    T-process quadrature filter and smoother for the Student's t inference. Uses RQ kernel and fully-symmetric
    point-sets by default. RQ kernel expectations w.r.t. Student's t-density are expressed as a simplified scale
    mixture representation which facilitates analytical tractability.
    """

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, kernel='rq', points='fs', point_hyp=None, dof=3.0):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = TPQ(nq, kern_par_dyn, kernel, points, point_hyp, dof)
        t_obs = TPQ(nr, kern_par_obs, kernel, points, point_hyp, dof)
        super(TPQStudent, self).__init__(ssm, t_dyn, t_obs)


def main():
    # UNGM demo
    from models.ungm import ungm_filter_demo
    from inference.gpquad import GPQKalman
    ker_par = np.array([[1, 1, 0.3]])
    ker_par_rbf = np.array([[1, 0.3]])
    pts_par = {'kappa': 0.0}
    ungm_filter_demo(TPQStudent, ker_par, ker_par, kernel='rq', points='fs', point_hyp=pts_par)
    # ungm_filter_demo(TPQKalman, ker_par_rbf, ker_par_rbf, kernel='rbf', points='ut', point_hyp=pts_par)
    # ungm_filter_demo(GPQKalman, ker_par_rbf, ker_par_rbf, kernel='rbf', points='ut', point_hyp=pts_par)


if __name__ == '__main__':
    main()
