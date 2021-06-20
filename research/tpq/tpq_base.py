import numpy as np
from numpy import newaxis as na
from ssmtoybox.mtran import LinearizationTransform, FullySymmetricStudentTransform
from ssmtoybox.bq.bqmtran import GaussianProcessTransform, StudentTProcessTransform, BQTransform
from ssmtoybox.bq.bqkern import RBFStudent
from ssmtoybox.ssmod import TransitionModel, MeasurementModel, UNGMTransition, UNGMMeasurement
from ssmtoybox.ssinf import StudentianInference
from ssmtoybox.utils import log_cred_ratio, mse_matrix, gauss_mixture, multivariate_t, RandomVariable, GaussRV, \
    StudentRV


# Gaussian mixture random variable
class GaussianMixtureRV(RandomVariable):

    def __init__(self, dim, means, covs, alphas):
        if len(means) != len(covs) != len(alphas):
            raise ValueError('Same number of means, covariances and mixture weights needs to be supplied!')

        if not np.isclose(np.sum(alphas), 1.0):
            ValueError('Mixture weights must sum to unity!')

        self.dim = dim
        self.means = means
        self.covs = covs
        self.alphas = alphas

    def sample(self, size):
        return np.moveaxis(gauss_mixture(self.means, self.covs, self.alphas, size), -1, 0)

    def get_stats(self):
        return self.means, self.covs, self.alphas


# Student's t-filters
class ExtendedStudent(StudentianInference):

    def __init__(self, dyn, obs, dof=4.0, fixed_dof=True):
        tf = LinearizationTransform(dyn.dim_in)
        th = LinearizationTransform(obs.dim_in)
        super(ExtendedStudent, self).__init__(dyn, obs, tf, th, dof, fixed_dof)


class GPQStudent(StudentianInference):

    def __init__(self, dyn, obs, kern_par_dyn, kern_par_obs, point_hyp=None, dof=4.0, fixed_dof=True):
        """
        Student filter with Gaussian Process quadrature moment transforms using fully-symmetric sigma-point set.

        Parameters
        ----------
        dyn : TransitionModel

        obs : MeasurementModel

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

        # degrees of freedom for SSM noises
        _, _, q_dof = dyn.noise_rv.get_stats()
        _, _, r_dof = obs.noise_rv.get_stats()

        # add DOF of the noises to the sigma-point parameters
        if point_hyp is None:
                point_hyp = dict()
        point_hyp_dyn = point_hyp
        point_hyp_obs = point_hyp
        point_hyp_dyn.update({'dof': q_dof})
        point_hyp_obs.update({'dof': r_dof})

        # init moment transforms
        t_dyn = GaussianProcessTransform(dyn.dim_in, kern_par_dyn, 'rbf-student', 'fs', point_hyp_dyn)
        t_obs = GaussianProcessTransform(obs.dim_in, kern_par_obs, 'rbf-student', 'fs', point_hyp_obs)
        super(GPQStudent, self).__init__(dyn, obs, t_dyn, t_obs, dof, fixed_dof)


class FSQStudent(StudentianInference):
    """Filter based on fully symmetric quadrature rules."""

    def __init__(self, dyn, obs, degree=3, kappa=None, dof=4.0, fixed_dof=True):

        # degrees of freedom for SSM noises
        _, _, q_dof = dyn.noise_rv.get_stats()
        _, _, r_dof = obs.noise_rv.get_stats()

        # init moment transforms
        t_dyn = FullySymmetricStudentTransform(dyn.dim_in, degree, kappa, q_dof)
        t_obs = FullySymmetricStudentTransform(obs.dim_in, degree, kappa, r_dof)
        super(FSQStudent, self).__init__(dyn, obs, t_dyn, t_obs, dof, fixed_dof)


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
    xD = filters[0].mod_dyn.dim_state

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

