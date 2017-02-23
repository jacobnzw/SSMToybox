import numpy as np
from fusion_paper.figprint import *
from numpy import newaxis as na
from transforms.taylor import Taylor1stOrder
from transforms.quad import FullySymmetricStudent
from models.ssmodel import StateSpaceModel, StudentStateSpaceModel
from inference.tpquad import TPQKalman, TPQStudent
from inference.gpquad import GPQKalman, GPQ
from inference.ssinfer import StudentInference
from inference.unscented import UnscentedKalman
from inference.cubature import CubatureKalman
from transforms.bqkernel import RBFStudent
from transforms.bayesquad import BQTransform
from system.datagen import System
from utils import log_cred_ratio, mse_matrix, bigauss_mixture, multivariate_t


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


def rbf_student_mc_weights(x, kern, num_samples, num_batch):
    # MC approximated BQ weights using RBF kernel and Student density
    # MC computed by batches, because without batches we would run out of memory for large sample sizes

    assert isinstance(kern, RBFStudent)
    # kernel parameters and input dimensionality
    par = kern.par
    dim, num_pts = x.shape

    # inverse kernel matrix
    iK = kern.eval_inv_dot(kern.par, x, scaling=False)
    mean, scale, dof = np.zeros((dim, )), np.eye(dim), kern.dof

    # compute MC estimates by batches
    num_samples_batch = num_samples // num_batch
    q_batch = np.zeros((num_pts, num_batch, ))
    Q_batch = np.zeros((num_pts, num_pts, num_batch))
    R_batch = np.zeros((dim, num_pts, num_batch))
    for ib in range(num_batch):

        # multivariate t samples
        x_samples = multivariate_t(mean, scale, dof, num_samples_batch).T

        # evaluate kernel
        k_samples = kern.eval(par, x_samples, x, scaling=False)
        kk_samples = k_samples[:, na, :] * k_samples[..., na]
        xk_samples = x_samples[..., na] * k_samples[na, ...]

        # intermediate sums
        q_batch[..., ib] = k_samples.sum(axis=0)
        Q_batch[..., ib] = kk_samples.sum(axis=0)
        R_batch[..., ib] = xk_samples.sum(axis=1)

    # MC approximations == sum the sums divide by num_samples
    c = 1/num_samples
    q = c * q_batch.sum(axis=-1)
    Q = c * Q_batch.sum(axis=-1)
    R = c * R_batch.sum(axis=-1)

    # BQ moment transform weights
    wm = q.dot(iK)
    wc = iK.dot(Q).dot(iK)
    wcc = R.dot(iK)
    return wm, wc, wcc, Q


def eval_perf_scores(x, mf, Pf):
    xD, steps, mc_sims, num_filt = mf.shape

    # average RMSE over simulations
    rmse = np.sqrt(((x[..., na] - mf) ** 2).sum(axis=0))
    rmse_avg = rmse.mean(axis=1)

    reg = 1e-6 * np.eye(xD)

    # average inclination indicator over simulations
    lcr = np.empty((steps, mc_sims, num_filt))
    for f in range(num_filt):
        for k in range(steps):
            mse = mse_matrix(x[:, k, :], mf[:, k, :, f]) + reg
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
            'q_cov_1': 1 * np.eye(self.qD),
            'r_mean_0': np.zeros(self.rD),
            'r_mean_1': np.zeros(self.rD),
            'r_cov_0': 0.01 * np.eye(self.rD),
            'r_cov_1': 1 * np.eye(self.rD),
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
        return bigauss_mixture(m0, c0, m1, c1, 0.7, size)

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

    # load data from mat-file
    # from scipy.io import loadmat
    # datadict = loadmat('synth_data', variable_names=('x', 'y'))
    # x, z = datadict['x'][:, 1:, :], datadict['y'][:, 1:, :]

    # init SSM for the filter
    ssm = SyntheticSSM()

    # kernel parameters for TPQ and GPQ filters
    # TPQ Student
    a, b = 10, 30
    par_dyn_tp = np.array([[0.4, a, a]])
    par_obs_tp = np.array([[0.4, b, b, b, b]])
    # par_dyn_tp = np.array([[1.0, 1.7, 1.7]])
    # par_obs_tp = np.array([[1.1, 3.0, 3.0, 3.0, 3.0]])
    # GPQ Student
    par_dyn_gpqs = np.array([[1.0, 5, 5]])
    par_obs_gpqs = np.array([[0.9, 4, 4, 4, 4]])
    # GPQ Kalman
    par_dyn_gpqk = np.array([[1.0, 2.0, 2.0]])
    par_obs_gpqk = np.array([[1.0, 2.0, 2.0, 2.0, 2.0]])
    # parameters of the point-set
    par_pt = {'kappa': None}

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        FSQStudent(ssm, kappa=None),
        # UnscentedKalman(ssm, kappa=-1),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, kernel='rbf-student', dof=4.0, dof_tp=4.0, point_hyp=par_pt),
        # GPQStudent(ssm, par_dyn_gpqs, par_obs_gpqs),
        # TPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
        # GPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
    )

    # assign weights approximated by MC with lots of samples
    # very dirty code
    pts = filters[1].tf_dyn.model.points
    kern = filters[1].tf_dyn.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_dyn, BQTransform):
            f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
            f.tf_dyn.Q = Q
    pts = filters[1].tf_meas.model.points
    kern = filters[1].tf_meas.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_meas, BQTransform):
            f.tf_meas.wm, f.tf_meas.Wc, f.tf_meas.Wcc = wm, wc, wcc
            f.tf_meas.Q = Q

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


class UNGMSys(StateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1

    q_additive = True
    r_additive = True

    def __init__(self):
        pars = {
            'x0_mean': np.atleast_1d(0.0),
            'x0_cov': np.atleast_2d(1.0),
            'q_mean_0': np.zeros(self.qD),
            'q_mean_1': np.zeros(self.qD),
            'q_cov_0': 10 * np.eye(self.qD),
            'q_cov_1': 100 * np.eye(self.qD),
            'r_mean_0': np.zeros(self.rD),
            'r_mean_1': np.zeros(self.rD),
            'r_cov_0': 0.01 * np.eye(self.rD),
            'r_cov_1': 1 * np.eye(self.rD),
        }
        super(UNGMSys, self).__init__(**pars)

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * x[0] ** 2]) + r

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

        return bigauss_mixture(m0, c0, m1, c1, 0.8, size)


class UNGM(StudentStateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1

    q_additive = True
    r_additive = True

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        super(UNGM, self).__init__(**kwargs)
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
        super(UNGM, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * x[0] ** 2]) + r

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * pars[0])])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * r[0] * x[0], 0.05 * x[0] ** 2])


def ungm_demo(steps=250, mc_sims=50):
    sys = UNGMSys()
    x, z = sys.simulate(steps, mc_sims)

    # SSM noise covariances should follow the system
    ssm = UNGM(q_cov=0.1, r_cov=0.01)

    # kernel parameters for TPQ and GPQ filters
    # TPQ Student
    # par_dyn_tp = np.array([[1.8, 3.0]])
    # par_obs_tp = np.array([[0.4, 1.0, 1.0]])
    par_dyn_tp = np.array([[1.0, 1.0]])
    par_obs_tp = np.array([[1.0, 1.0]])
    # GPQ Student
    par_dyn_gpqs = np.array([[1.0, 0.5]])
    par_obs_gpqs = np.array([[1.0, 1, 10]])
    # GPQ Kalman
    par_dyn_gpqk = np.array([[1.0, 0.5]])
    par_obs_gpqk = np.array([[1.0, 1, 10]])
    # parameters of the point-set
    par_pt = {'kappa': None}

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        UnscentedKalman(ssm, kappa=None),
        FSQStudent(ssm, kappa=None),  # crashes, not necessarily a bug
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, kernel='rbf-student', dof=4.0, dof_tp=4.0, point_hyp=par_pt),
        # GPQStudent(ssm, par_dyn_gpqs, par_obs_gpqs),
        # TPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
        # GPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
    )
    itpq = np.argwhere([isinstance(filters[i], TPQStudent) for i in range(len(filters))]).squeeze()

    # assign weights approximated by MC with lots of samples
    # very dirty code
    pts = filters[itpq].tf_dyn.model.points
    kern = filters[itpq].tf_dyn.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_dyn, BQTransform):
            f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
            f.tf_dyn.Q = Q
    pts = filters[itpq].tf_meas.model.points
    kern = filters[itpq].tf_meas.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_meas, BQTransform):
            f.tf_meas.wm, f.tf_meas.Wc, f.tf_meas.Wcc = wm, wc, wcc
            f.tf_meas.Q = Q

    mf, Pf = run_filters(filters, z)

    rmse_avg, lcr_avg = eval_perf_scores(x, mf, Pf)

    # print out table
    import pandas as pd
    pd.set_option('display.precision', 4)
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'MAX_RMSE', 'MEAN_INC', 'MAX_INC']
    data = np.array([rmse_avg.mean(axis=0), rmse_avg.max(axis=0), lcr_avg.mean(axis=0), lcr_avg.max(axis=0)]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # print kernel parameters
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(1)]
    partable = pd.DataFrame(np.vstack((par_dyn_tp, par_obs_tp)), columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)

    # plots
    fp = FigurePrint()
    time = np.arange(1, steps+1)

    # RMSE and INC box plots
    fig, ax = plt.subplots()
    ax.boxplot(rmse_avg, labels=['UKF', 'SF', 'TPQSF'])
    ax.set_ylabel('Average RMSE')
    plt.tight_layout(pad=0.1)
    fp.savefig('ungm_rmse_boxplot')

    fig, ax = plt.subplots()
    ax.boxplot(lcr_avg, labels=['UKF', 'SF', 'TPQSF'])
    ax.set_ylabel('Average INC')
    plt.tight_layout(pad=0.1)
    fp.savefig('ungm_inc_boxplot')

    # filtered state and covariance
    fig, ax = plt.subplots(3, 1, sharex=True)
    for fi, f in enumerate(filters):
        # true state
        ax[fi].plot(time, x[0, :, 0], 'r--', alpha=0.5)

        # measurements
        ax[fi].plot(time, z[0, :, 0], 'k.')

        xhat = mf[0, :, 0, fi]
        std = np.sqrt(Pf[0, 0, :, 0, fi])
        ax[fi].plot(time, xhat, label=f.__class__.__name__)
        ax[fi].fill_between(time, xhat - 2 * std, xhat + 2 * std, alpha=0.15)
        ax[fi].axis([None, None, -50, 50])
        ax[fi].legend()
    plt.show()

    # compare posterior variances with outliers
    plt.figure()
    plt.plot(time, z[0, :, 0], 'k.')
    for fi, f in enumerate(filters):
        plt.plot(time, 2*np.sqrt(Pf[0, 0, :, 0, fi]), label=f.__class__.__name__)
    plt.legend()
    plt.show()


class ReentryRadarSimpleSys(System):
    """
    Radar tracking of the reentry vehicle as described in [1]_.
    High velocity projectile is entering atmosphere, radar positioned 100,000ft above Earth's surface (and 100,
    000ft horizontally) is producing range measurements.

    State
    -----
    [p, v, x5]
    p - altitude,
    v - velocity,
    x5 - aerodynamic parameter

    Measurements
    ------------
    range and bearing


    References
    ----------
    .. [1] S. J. Julier, J. K. Uhlmann, and H. F. Durrant-Whyte, "A New Method for the Nonlinear Transformation
    of Means and Covariances in Filters and Estimators," IEEE Trans. Automat. Contr., vol. 45, no. 3,
    pp. 477–482, 2000.

    """

    xD = 3
    zD = 1  # measurement dimension
    qD = 3
    rD = 1  # measurement noise dimension
    q_additive = True
    r_additive = True

    R0 = 6371  # Earth's radius [km]  #2.0925e7  # Earth's radius [ft]
    # radar location: 30km (~100k ft) above the surface, radar-to-body horizontal range
    sx, sy = 30, 30
    Gamma = 1 / 6.096

    def __init__(self):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
        kwargs = {
            'x0_mean': np.array([90, 6, 1.5]),  # km, km/s
            'x0_cov': np.diag([0.3048 ** 2, 1.2192 ** 2, 1e-4]),  # km^2, km^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': 0.1 * np.eye(self.qD),
            'r_mean_0': np.zeros(self.rD),
            'r_cov_0': np.atleast_2d(0.03048 ** 2),
            'r_mean_1': np.zeros(self.rD),
            'r_cov_1': np.atleast_2d(0.1),
            'q_factor': np.eye(3),
        }
        super(ReentryRadarSimpleSys, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([-x[1] + q[0],
                         -np.exp(-self.Gamma * x[0]) * x[1] ** 2 * x[2] + q[1],
                         q[2]])

    def meas_fcn(self, x, r, pars):
        # range
        rng = np.sqrt(self.sx ** 2 + (x[0] - self.sy) ** 2)
        return np.array([rng]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def initial_condition_sample(self, size=None):
        m0, c0 = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(m0, c0, size).T

    def state_noise_sample(self, size=None):
        m0, c0 = self.get_pars('q_mean', 'q_cov')
        return np.random.multivariate_normal(m0, c0, size).T

    def measurement_noise_sample(self, size=None):
        m0, c0, m1, c1 = self.get_pars('r_mean_0', 'r_cov_0', 'r_mean_1', 'r_cov_1')
        return bigauss_mixture(m0, c0, m1, c1, 0.95, size)


class ReentryRadarSimple(StudentStateSpaceModel):
    """
    Radar tracking of the reentry vehicle as described in [1]_.
    Vehicle is entering Earth's atmosphere at high altitude and with great speed, ground radar is tracking it.

    State
    -----
    [px, py, vx, vy, x5]
    (px, py) - position,
    (vx, vy) - velocity,
    x5 - aerodynamic parameter

    Measurements
    ------------
    range and bearing


    References
    ----------
    .. [1] Julier, S. J., & Uhlmann, J. K. (2004). Unscented Filtering and Nonlinear Estimation.
           Proceedings of the IEEE, 92(3), 401-422

    """

    xD = 3
    zD = 1  # measurement dimension
    qD = 3
    rD = 1  # measurement noise dimension
    q_additive = True
    r_additive = True

    R0 = 6371  # Earth's radius [km]  #2.0925e7  # Earth's radius [ft]
    # radar location: 30km (~100k ft) above the surface, radar-to-body horizontal range
    sx, sy = 30, 30
    Gamma = 1 / 6.096

    def __init__(self, dt=0.1):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
        self.dt = dt
        kwargs = {
            'x0_mean': np.array([90, 6, 1.7]),  # km, km/s
            'x0_cov': np.diag([0.3048 ** 2, 1.2192 ** 2, 10]),  # km^2, km^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': 0.1 * np.eye(self.qD),
            'q_dof': 4.0,
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[0.03048 ** 2]]),
            'r_dof': 4.0,
            'q_gain': np.eye(3),
        }
        super(ReentryRadarSimple, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] - self.dt * x[1] + q[0],
                         x[1] - self.dt * np.exp(-self.Gamma * x[0]) * x[1] ** 2 * x[2] + q[1],
                         x[2] + q[2]])

    def meas_fcn(self, x, r, pars):
        # range
        rng = np.sqrt(self.sx ** 2 + (x[0] - self.sy) ** 2)
        return np.array([rng]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


def reentry_tracking_demo(mc_sims=100):

    # generate some data (true trajectory)
    sys = ReentryRadarSimpleSys()
    x = sys.simulate_trajectory(method='rk4', dt=0.1, duration=30, mc_sims=mc_sims)

    # pick only non-divergent trajectories
    x = x[..., np.all(x >= 0, axis=(0, 1))]
    mc = x.shape[2]

    z = np.zeros((sys.zD,) + x.shape[1:])
    for i in range(mc):
        z[..., i] = sys.simulate_measurements(x[..., i]).squeeze()

    # init SSM for the filters
    ssm = ReentryRadarSimple(dt=0.1)

    par_dyn_tp = np.array([[0.5, 10, 10, 10]])
    par_obs_tp = np.array([[0.5, 15, 20, 20]])
    par_dyn_gpqk = par_dyn_tp
    par_obs_gpqk = par_obs_tp
    par_pt = {'kappa': None}

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        FSQStudent(ssm, kappa=None),  # crashes, not necessarily a bug
        UnscentedKalman(ssm, kappa=None),
        # TPQStudent(ssm, par_dyn_tp, par_obs_tp, kernel='rbf-student', dof=4.0, dof_tp=4.0, point_hyp=par_pt),
        # GPQStudent(ssm, par_dyn_gpqs, par_obs_gpqs),
        # TPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='ut', point_hyp=par_pt),
        # GPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='ut', point_hyp=par_pt),
    )

    # assign weights approximated by MC with lots of samples
    # very dirty code
    # pts = filters[1].tf_dyn.model.points
    # kern = filters[1].tf_dyn.model.kernel
    # wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    # for f in filters:
    #     if isinstance(f.tf_dyn, BQTransform):
    #         f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
    #         f.tf_dyn.Q = Q
    # pts = filters[1].tf_meas.model.points
    # kern = filters[1].tf_meas.model.kernel
    # wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    # for f in filters:
    #     if isinstance(f.tf_meas, BQTransform):
    #         f.tf_meas.wm, f.tf_meas.Wc, f.tf_meas.Wcc = wm, wc, wcc
    #         f.tf_meas.Q = Q

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
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(sys.xD)]
    partable = pd.DataFrame(np.vstack((par_dyn_tp, par_obs_tp)), columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)


class CoordinatedTurnBOTSys(StateSpaceModel):
    """
    Bearings only target tracking in 2D using multiple sensors as in [3]_.

    TODO:
    Coordinated turn model [1]_ assumes constant turn rate (not implemented).
    Model in [2]_ is implemented here, where the turn rate can change in time and measurements are range and
    bearing.
    [3]_ considers only bearing measurements.

    State
    -----
    x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position [m]
        v_1, v_2 - target velocity [m/s]
        omega - target turn rate [deg/s]

    Measurements
    ------------


    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    xD = 5
    zD = 4  # measurement dimension == # sensors
    qD = 5
    rD = 4  # measurement noise dimension == # sensors

    q_additive = True
    r_additive = True

    rho_1, rho_2 = 0.1, 1.75e-4  # noise intensities

    def __init__(self, dt=0.1, sensor_pos=np.vstack((np.eye(2), -np.eye(2)))):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        sensor_pos :
            sensor [x, y] positions in rows
        """
        self.dt = dt
        self.sensor_pos = sensor_pos  # np.vstack((np.eye(2), -np.eye(2)))
        q_cov = np.array(
                [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
                 [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
                 [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
                 [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
                 [0, 0, 0, 0, self.rho_2 * self.dt]])
        r_cov = 10e-6 * np.eye(self.rD)
        kwargs = {
            'x0_mean': np.array([1000, 300, 1000, 0, -3.0 * np.pi / 180]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([100, 10, 100, 10, 10e-4]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean_0': np.zeros(self.qD),
            'q_cov_0': q_cov,
            'q_mean_1': np.zeros(self.qD),
            'q_cov_1': 50 * q_cov,
            'r_mean_0': np.zeros(self.rD),
            'r_cov_0': r_cov,  # 10e-3 rad == 10 mrad
            'r_mean_1': np.zeros(self.rD),
            'r_cov_1': 50 * r_cov,  # 10e-3 rad == 10 mrad
        }
        super(CoordinatedTurnBOTSys, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        """
        Model describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).

        Parameters
        ----------
        x
        q
        args


        Returns
        -------

        """
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om if om != 0 else self.dt
        d = (1 - np.cos(om * self.dt)) / om if om != 0 else 0
        mdyn = np.array([[1, c, 0, -d, 0],
                         [0, b, 0, -a, 0],
                         [0, d, 1, c, 0],
                         [0, a, 0, b, 0],
                         [0, 0, 0, 0, 1]])
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        """
        Bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x
        r
        args

        Returns
        -------

        """
        a = x[2] - self.sensor_pos[:, 1]
        b = x[0] - self.sensor_pos[:, 0]
        h = np.arctan(a / b)
        ig = h > 0.5*np.pi
        il = h < -0.5*np.pi
        if np.count_nonzero(il) > np.count_nonzero(ig):
            h[ig] -= 2*np.pi
        else:
            h[il] += 2*np.pi
        return h + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def initial_condition_sample(self, size=None):
        m0, c0 = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(m0, c0, size).T

    def state_noise_sample(self, size=None):
        m0, c0, m1, c1 = self.get_pars('q_mean_0', 'q_cov_0', 'q_mean_1', 'q_cov_1')
        return bigauss_mixture(m0, c0, m1, c1, 0.95, size)

    def measurement_noise_sample(self, size=None):
        m0, c0, m1, c1 = self.get_pars('r_mean_0', 'r_cov_0', 'r_mean_1', 'r_cov_1')
        return bigauss_mixture(m0, c0, m1, c1, 0.90, size)


class CoordinatedTurnBOT(StudentStateSpaceModel):
    """
    Bearings only target tracking in 2D using multiple sensors as in [3]_.

    TODO:
    Coordinated turn model [1]_ assumes constant turn rate (not implemented).
    Model in [2]_ is implemented here, where the turn rate can change in time and measurements are range and
    bearing.
    [3]_ considers only bearing measurements.

    State
    -----
    x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position [m]
        v_1, v_2 - target velocity [m/s]
        omega - target turn rate [deg/s]

    Measurements
    ------------


    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    xD = 5
    zD = 4  # measurement dimension == # sensors
    qD = 5
    rD = 4  # measurement noise dimension == # sensors

    q_additive = True
    r_additive = True

    rho_1, rho_2 = 0.1, 1.75e-4  # noise intensities

    def __init__(self, dt=0.1, sensor_pos=np.vstack((np.eye(2), -np.eye(2)))):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        sensor_pos :
            sensor [x, y] positions in rows
        """
        self.dt = dt
        self.sensor_pos = sensor_pos  # np.vstack((np.eye(2), -np.eye(2)))
        kwargs = {
            'x0_mean': np.array([1000, 300, 1000, 0, -3.0 * np.pi / 180]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([100, 10, 100, 10, 10e-4]),  # m^2, (m/s)^2 m^2, (m/s)^2, (rad/s)^2
            'x0_dof': 4.0,
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array(
                [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
                 [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
                 [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
                 [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
                 [0, 0, 0, 0, self.rho_2 * self.dt]]),
            'q_dof': 4.0,
            'r_mean': np.zeros(self.rD),
            'r_cov': 10e-6 * np.eye(self.rD),
            'r_dof': 4.0,
        }
        super(CoordinatedTurnBOT, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        """
        Model describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).

        Parameters
        ----------
        x
        q
        args


        Returns
        -------

        """
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om if om != 0 else self.dt
        d = (1 - np.cos(om * self.dt)) / om if om != 0 else 0
        mdyn = np.array([[1, c, 0, -d, 0],
                         [0, b, 0, -a, 0],
                         [0, d, 1, c, 0],
                         [0, a, 0, b, 0],
                         [0, 0, 0, 0, 1]])
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        """
        Bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x
        r
        args

        Returns
        -------

        """
        a = x[2] - self.sensor_pos[:, 1]
        b = x[0] - self.sensor_pos[:, 0]
        h = np.arctan(a / b)
        ig = h > 0.5*np.pi
        il = h < -0.5*np.pi
        if np.count_nonzero(il) > np.count_nonzero(ig):
            h[ig] -= 2*np.pi
        else:
            h[il] += 2*np.pi
        return h + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


def coordinated_bot_demo(steps=100, mc_sims=100):

    # sensor positions
    x_min, x_max = -10000, 10000
    y_min, y_max = -10000, 10000
    S = np.array([[x_min, y_min],
                  [x_min, y_max],
                  [x_max, y_min],
                  [x_max, y_max]])
    tau = 1.0
    # generate data
    sys = CoordinatedTurnBOTSys(dt=tau, sensor_pos=S)
    x, z = sys.simulate(steps, mc_sims)

    # weed out trajectories venturing outside of the sensor rectangle
    ix = np.all(np.abs(x[(0, 2), ...]) <= x_max, axis=(0, 1))
    x, z = x[..., ix], z[..., ix]
    print('{:.2f}% of trajectories weeded out.'.format(100 * np.count_nonzero(ix==False)/len(ix)))

    # SSM for the filters
    ssm = CoordinatedTurnBOT(dt=tau, sensor_pos=S)

    par_dyn_tp = np.array([[1.0, 1, 1, 1, 1, 1]])
    par_obs_tp = np.array([[1.0, 1, 100, 1, 100, 100]])
    par_dyn_gpqk = par_dyn_tp
    par_obs_gpqk = par_obs_tp
    par_pt = {'kappa': None}

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        FSQStudent(ssm, kappa=None),  # crashes, not necessarily a bug
        UnscentedKalman(ssm, kappa=None),
        # TPQStudent(ssm, par_dyn_tp, par_obs_tp, kernel='rbf-student', dof=4.0, dof_tp=4.0, point_hyp=par_pt),
        # GPQStudent(ssm, par_dyn_gpqs, par_obs_gpqs),
        # TPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='ut', point_hyp=par_pt),
        # GPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='ut', point_hyp=par_pt),
    )

    # assign weights approximated by MC with lots of samples
    # very dirty code
    # pts = filters[1].tf_dyn.model.points
    # kern = filters[1].tf_dyn.model.kernel
    # wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    # for f in filters:
    #     if isinstance(f.tf_dyn, BQTransform):
    #         f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
    #         f.tf_dyn.Q = Q
    # pts = filters[1].tf_meas.model.points
    # kern = filters[1].tf_meas.model.kernel
    # wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    # for f in filters:
    #     if isinstance(f.tf_meas, BQTransform):
    #         f.tf_meas.wm, f.tf_meas.Wc, f.tf_meas.Wcc = wm, wc, wcc
    #         f.tf_meas.Q = Q

    mf, Pf = run_filters(filters, z)

    # TODO: compare with results in "On Gaussian Optimal Smoothing of Non-Linear State Space Models"
    pos_x, pos_mf, pos_Pf = x[[0, 2], ...], mf[[0, 2], ...], Pf[np.ix_([0, 2], [0, 2])]
    vel_x, vel_mf, vel_Pf = x[[1, 3], ...], mf[[1, 3], ...], Pf[np.ix_([1, 3], [1, 3])]
    ome_x, ome_mf, ome_Pf = x[4, na, ...], mf[4, na, ...], Pf[4, 4, na, na, ...]
    pos_rmse, pos_lcr = eval_perf_scores(pos_x, pos_mf, pos_Pf)
    vel_rmse, vel_lcr = eval_perf_scores(vel_x, vel_mf, vel_Pf)
    ome_rmse, ome_lcr = eval_perf_scores(ome_x, ome_mf, ome_Pf)

    # print out table
    import pandas as pd
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'MAX_RMSE', 'MEAN_INC', 'MAX_INC']

    pos_data = np.array([pos_rmse.mean(axis=0), pos_rmse.max(axis=0), pos_lcr.mean(axis=0), pos_lcr.max(axis=0)]).T
    vel_data = np.array([vel_rmse.mean(axis=0), vel_rmse.max(axis=0), vel_lcr.mean(axis=0), vel_lcr.max(axis=0)]).T
    ome_data = np.array([ome_rmse.mean(axis=0), ome_rmse.max(axis=0), ome_lcr.mean(axis=0), ome_lcr.max(axis=0)]).T

    pos_table = pd.DataFrame(pos_data, f_label, m_label)
    vel_table = pd.DataFrame(vel_data, f_label, m_label)
    ome_table = pd.DataFrame(ome_data, f_label, m_label)
    print(pos_table)
    print(vel_table)
    print(ome_table)

    # print kernel parameters
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(sys.xD)]
    partable = pd.DataFrame(np.vstack((par_dyn_tp, par_obs_tp)), columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)


class CoordinatedTurnRadarSys(StateSpaceModel):
    """
    Maneuvering target tracking using radar measurements .

    TODO:
    Coordinated turn model [1]_ assumes constant turn rate (not implemented).
    Model in [2]_ is implemented here, where the turn rate can change in time and measurements are range and bearing.
    [3]_ considers only bearing measurements.

    State
    -----
    x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position [m]
        v_1, v_2 - target velocity [m/s]
        omega - target turn rate [deg/s]

    Measurements
    ------------


    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    xD = 5
    zD = 2  # measurement dimension == # sensors
    qD = 5
    rD = 2  # measurement noise dimension == # sensors

    q_additive = True
    r_additive = True

    rho_1, rho_2 = 0.1, 1.75e-4  # noise intensities

    def __init__(self, dt=0.1):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
        self.dt = dt
        q_cov = np.array(
            [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
             [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
             [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
             [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
             [0, 0, 0, 0, self.rho_2 * self.dt]])
        r_cov = np.diag([100, 10e-6])
        kwargs = {
            'x0_mean': np.array([1000, 300, 300, 0, -1.0 * np.pi / 180]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([100, 10, 100, 10, 10e-4]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean_0': np.zeros(self.qD),
            'q_cov_0': q_cov,
            'q_mean_1': np.zeros(self.qD),
            'q_cov_1': 50 * q_cov,
            'r_mean_0': np.zeros(self.rD),
            'r_cov_0': r_cov,  # 10e-3 rad == 10 mrad
            'r_mean_1': np.zeros(self.rD),
            'r_cov_1': 50 * r_cov,  # 10e-3 rad == 10 mrad
        }
        super(CoordinatedTurnRadarSys, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        """
        Model describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).

        Parameters
        ----------
        x
        q
        args


        Returns
        -------

        """
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om if om != 0 else self.dt
        d = (1 - np.cos(om * self.dt)) / om if om != 0 else 0
        mdyn = np.array([[1, c, 0, -d, 0],
                         [0, b, 0, -a, 0],
                         [0, d, 1, c, 0],
                         [0, a, 0, b, 0],
                         [0, 0, 0, 0, 1]])
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        """
        Range and bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x
        r
        args

        Returns
        -------

        """
        rang = np.sqrt(x[0] ** 2 + x[2] ** 2)
        theta = np.arctan2(x[2], x[0])
        return np.asarray([rang, theta]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def initial_condition_sample(self, size=None):
        m0, c0 = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(m0, c0, size).T

    def state_noise_sample(self, size=None):
        m0, c0, m1, c1 = self.get_pars('q_mean_0', 'q_cov_0', 'q_mean_1', 'q_cov_1')
        return bigauss_mixture(m0, c0, m1, c1, 0.95, size)

    def measurement_noise_sample(self, size=None):
        m0, c0, m1, c1 = self.get_pars('r_mean_0', 'r_cov_0', 'r_mean_1', 'r_cov_1')
        return bigauss_mixture(m0, c0, m1, c1, 0.95, size)


class CoordinatedTurnRadar(StudentStateSpaceModel):
    """
    Maneuvering target tracking using radar measurements .

    TODO:
    Coordinated turn model [1]_ assumes constant turn rate (not implemented).
    Model in [2]_ is implemented here, where the turn rate can change in time and measurements are range and
    bearing.
    [3]_ considers only bearing measurements.

    State
    -----
    x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position [m]
        v_1, v_2 - target velocity [m/s]
        omega - target turn rate [deg/s]

    Measurements
    ------------


    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    xD = 5
    zD = 2  # measurement dimension == # sensors
    qD = 5
    rD = 2  # measurement noise dimension == # sensors

    q_additive = True
    r_additive = True

    rho_1, rho_2 = 0.1, 1.75e-4  # noise intensities

    def __init__(self, dt=0.1):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
        self.dt = dt
        q_cov = np.array(
            [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
             [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
             [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
             [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
             [0, 0, 0, 0, self.rho_2 * self.dt]])
        r_cov = np.diag([100, 10e-6])
        kwargs = {
            'x0_mean': np.array([1000, 300, 300, 0, -1.0 * np.pi / 180]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([100, 10, 100, 10, 10e-4]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'x0_dof': 4.0,
            'q_mean': np.zeros(self.qD),
            'q_cov': q_cov,
            'q_dof': 4.0,
            'r_mean': np.zeros(self.rD),
            'r_cov': r_cov,
            'r_dof': 4.0,
        }
        super(CoordinatedTurnRadar, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        """
        Model describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).

        Parameters
        ----------
        x
        q
        args


        Returns
        -------

        """
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om if om != 0 else self.dt
        d = (1 - np.cos(om * self.dt)) / om if om != 0 else 0
        mdyn = np.array([[1, c, 0, -d, 0],
                         [0, b, 0, -a, 0],
                         [0, d, 1, c, 0],
                         [0, a, 0, b, 0],
                         [0, 0, 0, 0, 1]])
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        """
        Range and bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x
        r
        args

        Returns
        -------

        """
        rang = np.sqrt(x[0] ** 2 + x[2] ** 2)
        theta = np.arctan2(x[2], x[0])
        return np.asarray([rang, theta]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


def coordinated_radar_demo(steps=100, mc_sims=100, plots=True):
    tau = 1.0
    # generate data
    sys = CoordinatedTurnRadarSys(dt=tau)
    x, z = sys.simulate(steps, mc_sims)

    # weed out trajectories outside 10km radius
    # ix = np.all(np.linalg.norm(x[(0, 2), ...], axis=0, keepdims=True) <= 10000, axis=(0, 1))
    # x, z = x[..., ix], z[..., ix]
    # print('{:.2f}% of trajectories weeded out.'.format(100 * np.count_nonzero(ix==False) / len(ix)))

    if plots:
        plt.figure()
        plt.plot([0], [0], 'ko', label='radar')
        for i in range(x.shape[2]):
            plt.plot(x[0, :, i], x[2, :, i], 'b', alpha=0.1)
        plt.legend()
        plt.show()

    # SSM for the filters
    ssm = CoordinatedTurnRadar(dt=tau)

    # print initial conditions
    print('SSM x0_mean = {}'.format(ssm.get_pars('x0_mean')[0]))

    a = 1.0
    par_dyn_tp = np.array([[1.0, 1, a, 1, a, 1]])
    par_obs_tp = np.array([[1.0, 1, 1e2, 1, 1e2, 1e2]])
    par_dyn_gpqk = par_dyn_tp
    par_obs_gpqk = par_obs_tp
    par_pt = {'kappa': None}

    # print kernel parameters
    import pandas as pd
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(sys.xD)]
    partable = pd.DataFrame(np.vstack((par_dyn_tp, par_obs_tp)), columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)

    # init filters
    filters = (
        # ExtendedStudent(ssm),
        FSQStudent(ssm, kappa=None),
        # CubatureKalman(ssm),
        # UnscentedKalman(ssm, kappa=None),
        TPQStudent(ssm, par_dyn_tp, par_obs_tp, kernel='rbf-student', dof=4.0, dof_tp=4.0, point_hyp=par_pt),
        # GPQStudent(ssm, par_dyn_gpqs, par_obs_gpqs),
        # TPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='ut', point_hyp=par_pt),
        # GPQKalman(ssm, par_dyn_gpqk, par_obs_gpqk, points='ut', point_hyp=par_pt),
    )
    itpq = np.argwhere([isinstance(filters[i], TPQStudent) for i in range(len(filters))]).squeeze()

    # assign weights approximated by MC with lots of samples
    # very dirty code
    pts = filters[itpq].tf_dyn.model.points
    kern = filters[itpq].tf_dyn.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_dyn, BQTransform):
            f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
            f.tf_dyn.Q = Q
    pts = filters[itpq].tf_meas.model.points
    kern = filters[itpq].tf_meas.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_meas, BQTransform):
            f.tf_meas.wm, f.tf_meas.Wc, f.tf_meas.Wcc = wm, wc, wcc
            f.tf_meas.Q = Q

    mf, Pf = run_filters(filters, z)

    pos_x, pos_mf, pos_Pf = x[[0, 2], ...], mf[[0, 2], ...], Pf[np.ix_([0, 2], [0, 2])]
    vel_x, vel_mf, vel_Pf = x[[1, 3], ...], mf[[1, 3], ...], Pf[np.ix_([1, 3], [1, 3])]
    ome_x, ome_mf, ome_Pf = x[4, na, ...], mf[4, na, ...], Pf[4, 4, na, na, ...]
    pos_rmse, pos_lcr = eval_perf_scores(pos_x, pos_mf, pos_Pf)
    vel_rmse, vel_lcr = eval_perf_scores(vel_x, vel_mf, vel_Pf)
    ome_rmse, ome_lcr = eval_perf_scores(ome_x, ome_mf, ome_Pf)

    # plot metrics
    time = np.arange(1, steps+1)
    fig, ax = plt.subplots(3, 1, sharex=True)
    for fi, f in enumerate(filters):
        ax[0].semilogy(time, pos_rmse[..., fi], label=f.__class__.__name__)
        ax[1].semilogy(time, vel_rmse[..., fi], label=f.__class__.__name__)
        ax[2].semilogy(time, ome_rmse[..., fi], label=f.__class__.__name__)
    plt.legend()
    plt.show()

    # print out table
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'MAX_RMSE', 'MEAN_INC', 'MAX_INC']

    pos_data = np.array([pos_rmse.mean(axis=0), pos_rmse.max(axis=0), pos_lcr.mean(axis=0), pos_lcr.max(axis=0)]).T
    vel_data = np.array([vel_rmse.mean(axis=0), vel_rmse.max(axis=0), vel_lcr.mean(axis=0), vel_lcr.max(axis=0)]).T
    ome_data = np.array([ome_rmse.mean(axis=0), ome_rmse.max(axis=0), ome_lcr.mean(axis=0), ome_lcr.max(axis=0)]).T

    pos_table = pd.DataFrame(pos_data, f_label, m_label)
    vel_table = pd.DataFrame(vel_data, f_label, m_label)
    ome_table = pd.DataFrame(ome_data, f_label, m_label)
    print(pos_table)
    print(vel_table)
    print(ome_table)


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    # synthetic_demo(mc_sims=50)
    ungm_demo()
    # reentry_tracking_demo(mc_sims=50)
    # coordinated_bot_demo(steps=40, mc_sims=100)
    # coordinated_radar_demo(steps=100, mc_sims=100, plots=False)