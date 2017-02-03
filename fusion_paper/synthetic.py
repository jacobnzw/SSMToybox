import numpy as np
from numpy import newaxis as na
from datagen import System
from transforms.taylor import Taylor1stOrder
from transforms.quad import FullySymmetricStudent
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
        super(SyntheticSys, self).__init__(**pars)

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
        q0 = np.random.multivariate_normal(m0, c0, size).T
        q1 = np.random.multivariate_normal(m1, c1, size).T
        return 0.95 * q0 + 0.05 * q1

    def measurement_noise_sample(self, size=None):
        m0, m1, c0, c1 = self.get_pars('r_mean_0', 'r_mean_1', 'r_cov_0', 'r_cov_1')
        r0 = np.random.multivariate_normal(m0, c0, size).T
        r1 = np.random.multivariate_normal(m1, c1, size).T
        return 0.9 * r0 + 0.1 * r1

    def initial_condition_sample(self, size=None):
        m, c = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(m, c, size).T


class SyntheticSSM(StudentStateSpaceModel):

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

    def __init__(self, sys, dof=4.0, fixed_dof=True):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        tf = Taylor1stOrder(nq)
        th = Taylor1stOrder(nr)
        super(ExtendedStudent, self).__init__(sys, tf, th, dof, fixed_dof)


class GPQStudent(StudentInference):

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, point_hyp=None, dof=4.0, fixed_dof=True):
        """
        Student filter with Gaussian Process quadrature moment transforms using fully-symmetric sigma-point set.

        Parameters
        ----------
        ssm : StudentStateSpaceModel
        kern_par_dyn : numpy.ndarray
            Kernel parameters for the GPQ moment transform of the dynamics.
        kern_par_obs : numpy.ndarray
            Kernel parameters for the GPQ moment transform of the measurement function.
        point_hyp : dict
            Point set parameters with keys:
              * `'degree'`: Degree (order) of the quadrature rule.
              * `'kappa'`: Tuning parameter of controlling spread of sigma-points around the center.
        dof : float
            Desired degree of freedom for the filtered density.
        fixed_dof : bool
            If `True`, DOF will be fixed for all time steps, which preserves the heavy-tailed behaviour of the filter.
            If `False`, DOF will be increasing after each measurement update, which means the heavy-tailed behaviour is
            not preserved and therefore converges to a Gaussian filter.
        """
        assert isinstance(ssm, StudentStateSpaceModel)

        # correct input dimension if noise non-additive
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD

        # degrees of freedom for SSM noises
        q_dof, r_dof = ssm.get_pars('q_dof', 'r_dof')

        # add DOF of the noises to the sigma-point parameters
        if point_hyp is None:
                point_hyp = dict()
        point_hyp_dyn = point_hyp
        point_hyp_obs = point_hyp
        point_hyp_dyn.update({'dof': q_dof})
        point_hyp_obs.update({'dof': r_dof})

        # init moment transforms
        t_dyn = GPQ(nq, kern_par_dyn, 'rq', 'fs', point_hyp_dyn)
        t_obs = GPQ(nr, kern_par_obs, 'rq', 'fs', point_hyp_obs)
        super(GPQStudent, self).__init__(ssm, t_dyn, t_obs, dof, fixed_dof)


class FSQStudent(StudentInference):
    """Filter based on fully symmetric quadrature rules."""

    def __init__(self, ssm, degree=3, kappa=None, dof=4.0, fixed_dof=True):
        assert isinstance(ssm, StudentStateSpaceModel)

        # correct input dimension if noise non-additive
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD

        # degrees of freedom for SSM noises
        q_dof, r_dof = ssm.get_pars('q_dof', 'r_dof')

        # init moment transforms
        t_dyn = FullySymmetricStudent(nq, degree, kappa, q_dof)
        t_obs = FullySymmetricStudent(nr, degree, kappa, r_dof)
        super(FSQStudent, self).__init__(ssm, t_dyn, t_obs, dof, fixed_dof)


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
    # TODO: Does the DOF of input density show up in kernel parameters?
    par_dyn = np.array([[1.0, 1.0, 3.0, 3.0]])
    par_obs = np.array([[1.0, 1.0, 3.0, 3.0, 3.0, 3.0]])

    # init filters
    # TODO: StudentSSM stores scale matrix parameter in *_cov variables, SPs created from cov = nu/(nu-2) * scale_matrix
    filters = (
        # ExtendedStudent(ssm),
        FSQStudent(ssm, kappa=-1),
        # UnscentedKalman(ssm, kappa=-1),
        # TPQStudent(ssm, par_dyn, par_obs, dof=4.0),
        # GPQStudent(ssm, par_dyn, par_obs),
    )
    num_filt = len(filters)

    # init space for filtered mean and covariance
    mf = np.zeros((ssm.xD, steps, mc_sims, num_filt))
    Pf = np.zeros((ssm.xD, ssm.xD, steps, mc_sims, num_filt))

    # run filters
    for i, f in enumerate(filters):
        print('Running {} ...'.format(f.__class__.__name__))
        for imc in range(mc_sims):
            mf[..., imc, i], Pf[..., imc, i] = f.forward_pass(z[..., imc])

    # evaluate performance metrics
    rmse = np.sqrt(((x[...,  na] - mf) ** 2).sum(axis=0))
    rmse_avg = rmse.mean(axis=1)  # average RMSE over simulations

    # print out table
    import pandas as pd
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'MAX_RMSE']
    data = np.array([rmse_avg.mean(axis=0), rmse_avg.max(axis=0)]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)


def synthetic_plots(steps=250, mc_sims=20):

    # generate data
    sys = SyntheticSys()
    x, z = sys.simulate(steps, mc_sims)

    import matplotlib.pyplot as plt
    for i in range(mc_sims):
        plt.plot(x[0, :, i], x[1, :, i], 'b', alpha=0.15)
    plt.show()


if __name__ == '__main__':
    synthetic_demo(mc_sims=500)
    # synthetic_plots()
