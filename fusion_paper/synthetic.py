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
from utils import log_cred_ratio, mse_matrix, bigauss_mixture


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
        return bigauss_mixture(m0, c0, m1, c1, 0.95, size)

    def measurement_noise_sample(self, size=None):
        m0, m1, c0, c1 = self.get_pars('r_mean_0', 'r_mean_1', 'r_cov_0', 'r_cov_1')
        return bigauss_mixture(m0, c0, m1, c1, 0.95, size)

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
            'q_cov': np.eye(self.qD),
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


class UNGMnonaddSys(StateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1

    q_additive = True
    r_additive = False

    def __init__(self):
        pars = {
            'x0_mean': np.atleast_1d(0.0),
            'x0_cov': np.atleast_2d(1.0),
            'q_mean_0': np.zeros(self.rD),
            'q_mean_1': np.zeros(self.rD),
            'q_cov_0': 0.01 * np.eye(self.qD),
            'q_cov_1': 5 * np.eye(self.qD),
            'r_mean_0': np.zeros(self.rD),
            'r_mean_1': np.zeros(self.rD),
            'r_cov_0': 0.01 * np.eye(self.rD),
            'r_cov_1': 5 * np.eye(self.rD),
        }
        super(UNGMnonaddSys, self).__init__(**pars)

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * r[0] * x[0] ** 2])

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * pars[0])])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * r[0] * x[0], 0.05 * x[0] ** 2])

    def initial_condition_sample(self, size=None):
        m, c = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(m, c, size).T

    def state_noise_sample(self, size=None):
        m0, c0 = self.get_pars('q_mean_0', 'q_cov_0')
        m1, c1 = self.get_pars('q_mean_1', 'q_cov_1')

        # samples from 2-component Gaussian mixture
        return bigauss_mixture(m0, c0, m1, c1, 0.95, size)

    def measurement_noise_sample(self, size=None):
        m0, c0 = self.get_pars('r_mean_0', 'r_cov_0')
        m1, c1 = self.get_pars('r_mean_1', 'r_cov_1')

        return bigauss_mixture(m0, c0, m1, c1, 0.9, size)


class UNGMnonadd(StudentStateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1

    q_additive = True
    r_additive = False

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        super(UNGMnonadd, self).__init__(**kwargs)
        kwargs = {
            'x0_mean': np.atleast_1d(x0_mean),
            'x0_cov': np.atleast_2d(x0_cov),
            'x0_dof': 4.0,
            'q_mean': np.atleast_1d(q_mean),
            'q_cov': np.atleast_2d(q_cov),
            'q_dof': 4.0,
            'r_mean': np.atleast_1d(r_mean),
            'r_cov': np.atleast_2d(r_cov),
            'r_dof': 4.0,
        }
        super(UNGMnonadd, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * r[0] * x[0] ** 2])

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * pars[0])])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * r[0] * x[0], 0.05 * x[0] ** 2])


# Student's t-filters
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
        t_dyn = GPQ(nq, kern_par_dyn, 'rbf-student', 'fs', point_hyp_dyn)
        t_obs = GPQ(nr, kern_par_obs, 'rbf-student', 'fs', point_hyp_obs)
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


def eval_perf_scores(x, mf, Pf):
    xD, steps, mc_sims, num_filt = mf.shape

    # average RMSE over simulations
    rmse = np.sqrt(((x[..., na] - mf) ** 2).sum(axis=0))
    rmse_avg = rmse.mean(axis=1)

    # average inclination indicator over simulations
    lcr = np.empty((steps, mc_sims, num_filt))
    for f in range(num_filt):
        for k in range(steps):
            mse = mse_matrix(x[:, k, :], mf[:, k, :, f])
            for imc in range(mc_sims):
                lcr[k, imc, f] = log_cred_ratio(x[:, k, imc], mf[:, k, imc, f], Pf[..., k, imc, f], mse)
    lcr_avg = lcr.mean(axis=1)

    return rmse_avg, lcr_avg


def run_filters(filters, z):
    num_filt = len(filters)
    zD, steps, mc_sims = z.shape
    xD = filters[0].ssm.xD

    # init space for filtered mean and covariance
    mf = np.zeros((xD, steps, mc_sims, num_filt))
    Pf = np.zeros((xD, xD, steps, mc_sims, num_filt))

    # run filters
    for i, f in enumerate(filters):
        print('Running {} ...'.format(f.__class__.__name__))
        for imc in range(mc_sims):
            mf[..., imc, i], Pf[..., imc, i] = f.forward_pass(z[..., imc])
            f.reset()

    # return filtered mean and covariance
    return mf, Pf


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
    # sys = SyntheticSys()
    # x, z = sys.simulate(steps, mc_sims)

    # load data from mat-file
    from scipy.io import loadmat
    datadict = loadmat('synth_data', variable_names=('x', 'y'))
    x, z = datadict['x'][:, 1:, :], datadict['y'][:, 1:, :]

    # init SSM for the filter
    ssm = SyntheticSSM()

    # kernel parameters for TPQ and GPQ filters
    # TPQ Student
    # par_dyn_tp = np.array([[1.0, 3.8, 3.8]])
    # par_obs_tp = np.array([[1.0, 4.0, 4.0, 4.0, 4.0]])
    par_dyn_tp = np.array([[1.0, 5, 5]])
    par_obs_tp = np.array([[0.9, 4.0, 4.0, 4.0, 4.0]])
    # GPQ Student
    par_dyn_gpqs = np.array([[1.0, 5, 5]])
    par_obs_gpqs = np.array([[0.9, 4, 4, 4, 4]])
    # GPQ Kalman
    par_dyn_gpqk = np.array([[1.0, 2.0, 2.0]])
    par_obs_gpqk = np.array([[1.0, 2.0, 2.0, 2.0, 2.0]])
    # parameters of the point-set
    par_pt = {'kappa': 1}

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        FSQStudent(ssm, kappa=1),
        # UnscentedKalman(ssm, kappa=-1),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, kernel='rbf-student', dof=4.0, dof_tp=4.0, point_hyp=par_pt),
        # GPQStudent(ssm, par_dyn_gpqs, par_obs_gpqs),
        # TPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
        # GPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
    )

    mf, Pf = run_filters(filters, z)

    lcr_avg, rmse_avg = eval_perf_scores(x, mf, Pf)

    # print out table
    import pandas as pd
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'MAX_RMSE', 'MEAN_INC', 'MAX_INC']
    data = np.array([rmse_avg.mean(axis=0), rmse_avg.max(axis=0), lcr_avg.mean(axis=0), lcr_avg.max(axis=0)]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # print kernel parameters
    parlab = ['alpha'] + ['ell_{}'.format(d+1) for d in range(4)]
    partable = pd.DataFrame(np.vstack((np.hstack((par_dyn_tp.squeeze(), np.zeros((2,)))), par_obs_tp)),
                            columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)


def synthetic_plots(steps=250, mc_sims=20):

    # generate data
    sys = SyntheticSys()
    x, z = sys.simulate(steps, mc_sims)

    import matplotlib.pyplot as plt
    for i in range(mc_sims):
        plt.plot(x[0, :, i], x[1, :, i], 'b', alpha=0.15)
    plt.show()


def ungm_demo(steps=250, mc_sims=100):
    sys = UNGMnonaddSys()
    x, z = sys.simulate(steps, mc_sims)

    # SSM noise covariances should follow the system
    ssm = UNGMnonadd(q_cov=0.01, r_cov=0.01)

    # kernel parameters for TPQ and GPQ filters
    # TPQ Student
    # par_dyn_tp = np.array([[1.0, 1.0, 0.8, 0.8]])
    # par_obs_tp = np.array([[1.0, 1.0, 1.1, 1.1, 1.1, 1.1]])
    par_dyn_tp = np.array([[1.0, 0.5]])
    par_obs_tp = np.array([[1.0, 1.0, 10.0]])
    # GPQ Student
    par_dyn_gpqs = np.array([[1.0, 0.5]])
    par_obs_gpqs = np.array([[1.0, 1, 10]])
    # GPQ Kalman
    par_dyn_gpqk = np.array([[1.0, 0.5]])
    par_obs_gpqk = np.array([[1.0, 1, 10]])
    # parameters of the point-set
    par_pt = {'kappa': 1}

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        # FSQStudent(ssm, kappa=None),  # crashes, not necessarily a bug
        # UnscentedKalman(ssm, kappa=None),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, kernel='rbf-student', dof=4.0, dof_tp=4.0, point_hyp=par_pt),
        # GPQStudent(ssm, par_dyn_gpqs, par_obs_gpqs),
        # TPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
        # GPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
    )

    mf, Pf = run_filters(filters, z)

    rmse_avg, lcr_avg = eval_perf_scores(x, mf, Pf)

    # print out table
    import pandas as pd
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'MAX_RMSE', 'MEAN_INC', 'MAX_INC']
    data = np.array([rmse_avg.mean(axis=0), rmse_avg.max(axis=0), lcr_avg.mean(axis=0), lcr_avg.max(axis=0)]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # print kernel parameters
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(2)]
    partable = pd.DataFrame(np.vstack((np.hstack((par_dyn_tp.squeeze(), np.zeros((1,)))), par_obs_tp)),
                            columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)

if __name__ == '__main__':
    # synthetic_demo(mc_sims=50)
    ungm_demo(mc_sims=50)