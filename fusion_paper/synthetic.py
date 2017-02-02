import numpy as np
from numpy import newaxis as na
from datagen import System
from transforms.taylor import Taylor1stOrder
from models.ssmodel import StateSpaceModel, StudentStateSpaceModel
from inference.tpquad import TPQKalman, TPQStudent
from inference.gpquad import GPQKalman, GPQ
from inference.ssinfer import StudentInference
from inference.unscented import UnscentedKalman


class SyntheticSys(StateSpaceModel):
    """
    Synthetic system from Filip Tronarp.
    """

    xD = 2
    zD = 2
    qD = 2
    rD = 2

    q_additive = True
    r_additive = False

    def __init__(self):
        pars = {
            'x0_mean': np.array([0.0, 50.0]),
            'x0_cov': 0.1 * np.eye(self.xD),
            'q_mean_0': np.zeros(self.qD),
            'q_mean_1': np.zeros(self.qD),
            'q_cov_0': 0.01 * np.eye(self.qD),
            'q_cov_1': 5 * np.eye(self.qD),
            'r_mean_0': np.zeros(self.rD),
            'r_mean_1': np.zeros(self.rD),
            'r_cov_0': 0.01 * np.eye(self.rD),
            'r_cov_1': 5 * np.eye(self.rD),
        }
        super(Synthetic, self).__init__(**pars)

    def dyn_fcn(self, x, q, pars):
        a = 1 - 0.1 / (1 + np.linalg.norm(x))
        b = 1 / (1 + np.linalg.norm(x))
        A = np.array([[a, b],
                      [-b, a]])
        return A.dot(x) + q

    def meas_fcn(self, x, r, pars):
        return (1 + r) * x

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def state_noise_sample(self, size=None):
        m0, m1, c0, c1 = self.get_pars('q_mean_0', 'q_mean_1', 'q_cov_0', 'q_cov_1')
        q0 = np.random.multivariate_normal(m0, c0, size)
        q1 = np.random.multivariate_normal(m1, c1, size)
        return 0.95 * q0 + 0.05 * q1

    def measurement_noise_sample(self, size=None):
        m0, m1, c0, c1 = self.get_pars('r_mean_0', 'r_mean_1', 'r_cov_0', 'r_cov_1')
        r0 = np.random.multivariate_normal(m0, c0, size)
        r1 = np.random.multivariate_normal(m1, c1, size)
        return 0.9 * r0 + 0.1 * r1

    def initial_condition_sample(self, size=None):
        m, c = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(m, c, size)


class SyntheticSSM(StudentStateSpaceModel):

    xD = 2
    zD = 2
    qD = 2
    rD = 2

    q_additive = True
    r_additive = False

    def __init__(self):
        pars = {
            'x0_mean': np.zeros(self.xD),
            'x0_cov': np.eye(self.xD),
            'q_mean': np.zeros(self.qD),
            'q_cov': 0.01 * np.eye(self.qD),
            'q_dof': 4.0,
            'r_mean': np.zeros(self.rD),
            'r_cov': 0.01 * np.eye(self.rD),
            'r_dof': 4.0,
        }
        super(SyntheticSSM, self).__init__(**pars)

    def dyn_fcn(self, x, q, pars):
        a = 1 - 0.1 / (1 + np.linalg.norm(x))
        b = 1 / (1 + np.linalg.norm(x))
        A = np.array([[a, b],
                      [-b, a]])
        return A.dot(x) + q

    def meas_fcn(self, x, r, pars):
        return (1 + r) * x

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def state_noise_sample(self, size=None):
        m, c, dof = self.get_pars('q_mean', 'q_cov', 'q_dof')
        return self._multivariate_t(m, c, dof, size)

    def measurement_noise_sample(self, size=None):
        m, c, dof = self.get_pars('r_mean', 'r_cov', 'r_dof')
        return self._multivariate_t(m, c, dof, size)

    def initial_condition_sample(self, size=None):
        m, c, dof = self.get_pars('x0_mean', 'x0_cov', )
        return np.random.multivariate_normal(m, c, size)


class ExtendedStudent(StudentInference):

    def __init__(self, sys):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Taylor1stOrder(nq)
        th = Taylor1stOrder(nr)
        super(ExtendedStudent, self).__init__(sys, tf, th)


class GPQStudent(StudentInference):

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, kernel='rq', points='fs', point_hyp=None):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = GPQ(nq, kern_par_dyn, kernel, points, point_hyp)
        t_obs = GPQ(nr, kern_par_obs, kernel, points, point_hyp)
        super(GPQStudent, self).__init__(ssm, t_dyn, t_obs)


def synthetic_demo(steps=250, mc_sims=5000):
    """
    An experiment replicating the conditions of the synthetic example in _[1] used for testing non-additive
    student's t sigma-point filters.

    Parameters
    ----------
    steps
    mc_sims

    Returns
    -------

    """

    # generate data
    sys = SyntheticSys()
    x, z = sys.simulate(steps, mc_sims)

    # init SSM for the filter
    ssm = SyntheticSSM()

    # kernel parameters for TPQ and GPQ filters
    par_dyn = np.array([[1.0, 1.0, 3.0]])
    par_obs = np.array([[1.0, 1.0, 3.0]])

    # init filters
    filters = (
        ExtendedStudent(ssm),
        UnscentedKalman(ssm),
        TPQStudent(ssm, par_dyn, par_obs, nu=4.0),
        GPQStudent(ssm, par_dyn, par_obs),
    )
    num_filt = len(filters)

    # init space for filtered mean and covariance
    mf = np.zeros((ssm.xD, steps, mc_sims, num_filt))
    Pf = np.zeros((ssm.xD, ssm.xD, steps, mc_sims, num_filt))

    # run filters
    for i, f in enumerate(filters):
        mf[..., i], Pf[..., i] = f.forward_pass(z)

    # evaluate performance metrics
    # FIXME: RMSE is a norm thus number -> sum out dimension
    rmse = np.sqrt((x[..., na] - mf) ** 2)
    rmse_avg = rmse.mean(axis=2)  # average RMSE over simulations

    # print out table
    import pandas as pd
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'MAX_RMSE']
    table = pd.DataFrame([rmse_avg.mean(axis=1), rmse_avg.max(axis=1)], f_label, m_label)