from inference.ssinfer import StateSpaceInference, MarginalInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import GPQ, GPQMO
import numpy as np
import pandas as pd


class GPQKalman(StateSpaceInference):
    """
    GP quadrature filter and smoother.
    """

    def __init__(self, ssm, kern_par_dyn, kern_par_obs, kernel='rbf', points='ut', point_hyp=None):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = GPQ(nq, kern_par_dyn, kernel, points, point_hyp)
        t_obs = GPQ(nr, kern_par_obs, kernel, points, point_hyp)
        super(GPQKalman, self).__init__(ssm, t_dyn, t_obs)


class GPQMOKalman(StateSpaceInference):

    def __init__(self, ssm, ker_par_dyn, ker_par_obs, kernel='rbf', points='ut', point_par=None):
        """
        Nonlinear Kalman filter based on Gaussian process quadrature with multiple independent outputs.

        Parameters
        ----------
        ssm : StateSpaceModel
        ker_par_dyn : numpy.ndarray
        ker_par_obs : numpy.ndarray
        kernel : string
        points : string
        point_par : dict
        """
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = GPQMO(nq, ssm.xD, ker_par_dyn, kernel, points, point_par)
        t_obs = GPQMO(nr, ssm.zD, ker_par_obs, kernel, points, point_par)
        super(GPQMOKalman, self).__init__(ssm, t_dyn, t_obs)


class GPQMKalman(MarginalInference):

    def __init__(self, sys, kernel, points, par_mean=None, par_cov=None, point_hyp=None):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        t_dyn = GPQ(nq, kernel, points, point_par=point_hyp)
        t_obs = GPQ(nr, kernel, points, point_par=point_hyp)
        super(GPQMKalman, self).__init__(sys, t_dyn, t_obs, par_mean, par_cov)


def main():
    # UNGM demo
    # These hyper-parameters provide visibly good GP fit (mean function follows the true function and predictive
    # variances are sufficiently large to cover the true function variation) for GH-15 points and YET the performance
    # of the filter is no better than the classical counterpart (GHKF-15). Whereas when the hyper-parameters for
    # the dynamics model are alpha=1.0, el=0.1 the GP fit looks completely ridiculous (overfit w/ low predictive
    # variance) and YET the filter outperforms the GHKF-15. How is this possible? Clearly the GP fit does not seem to
    # be decisive when it comes to filter performance.
    # from models.ungm import ungm_filter_demo
    # hdyn = {'alpha': 2.5, 'el': 0.1}
    # hobs = {'alpha': 1.0, 'el': 1.0 * np.ones(1)}
    # ut_hyp = {'kappa': 0.0}
    # ungm_filter_demo(GPQKalman, 'rbf', 'sr', kern_hyp_dyn=hdyn, kern_hyp_obs=hobs, point_hyp=ut_hyp)
    # par_prior_mean = np.log(np.array([1, 1, 1, 1]))
    # par_prior_cov = np.diag([1, 1, 1, 1])
    # ungm_filter_demo(GPQMKalman, 'rbf', 'sr', par_mean=par_prior_mean, par_cov=par_prior_cov)


    # Pendulum demo
    # from models.pendulum import pendulum_filter_demo
    # hdyn = {'alpha': 1.0, 'el': [3.0, 3.0]}
    # hobs = {'alpha': 1.0, 'el': [1.6, 1e5]}
    # gh_hyp = {'degree': 5}
    # kwargs_gh5 = {'kern_hyp_dyn': {'alpha': 1.0, 'el': [3.0, 3.0]},
    #               'kern_hyp_obs': {'alpha': 1.0, 'el': [2.0, 1e5]},
    #               'point_hyp': {'degree': 5}}
    # kwargs_ut = {'kern_hyp_dyn': {'alpha': 1.0, 'el': [3.0, 3.0]},  # emv 3.3, 5.9e4
    #              'kern_hyp_obs': {'alpha': 1.0, 'el': [2.0, 1e4]}, }
    # pendulum_filter_demo(GPQKalman, 'rbf', 'sr', **kwargs_ut)
    # pendulum_filter_demo(GPQMKalman, 'rbf', 'sr')


    # Reentry vehicle tracking demo
    # The radar measurement model only uses the first two state dimensions as input, which means that the remaining
    # state dimensions are irrelevant to the GP model of the measurement function. This is why we set high
    # lengthscale for the remaining dimensions, which ensures they will not contribute significantly to kernel
    # covariance (the RBF kernel is expressed in terms of inverse lengthscales).
    # TODO: find par that give position RMSE < ~0.01 (UKF), GPQKF best position RMSE is ~0.1,
    from models.tracking import reentry_filter_demo
    from unscented import UnscentedKalman
    dim = 5
    hdyn = np.array([[1, 25, 25, 25, 25, 25]], dtype=float)
    hobs = np.array([[1, 25, 25, 1e4, 1e4, 1e4]], dtype=float)
    dyn_par_table = pd.DataFrame(hdyn, columns=['alpha'] + ['el_' + str(d) for d in range(1, dim+1)])
    obs_par_table = pd.DataFrame(hobs, columns=['alpha'] + ['el_' + str(d) for d in range(1, dim+1)])
    print('Kernel parameters used')
    print(dyn_par_table)
    print(obs_par_table)
    reentry_filter_demo(GPQKalman, hdyn, hobs, 'rbf', 'ut')

    # multi-output GPQ Kalman
    null, lin = 1e4, 3
    # hdyn = np.array([[1, lin, null, lin, null, null],
    #                  [1, null, lin, null, lin, null],
    #                  [1, 3, null, 3, null, null],
    #                  [1, null, 3, null, 3, null],
    #                  [1, null, null, null, null, lin]], dtype=float)
    # hobs = np.array([[1, 3, 3, null, null, null],
    #                  [1, lin, lin, null, null, null]], dtype=float)
    hdyn = np.hstack((np.ones((5, 1)), 25.0*np.ones((5, 5))))
    hobs = np.array([[1, 25, 25, null, null, null],
                     [1, 25, 25, null, null, null]], dtype=float)
    dyn_par_table = pd.DataFrame(hdyn, columns=['alpha'] + ['el_' + str(d) for d in range(1, dim + 1)])
    obs_par_table = pd.DataFrame(hobs, columns=['alpha'] + ['el_' + str(d) for d in range(1, dim + 1)])
    print('Kernel parameters used')
    print(dyn_par_table)
    print(obs_par_table)
    reentry_filter_demo(GPQMOKalman, hdyn, hobs, 'rbf', 'ut')


    # reentry_filter_demo(UnscentedKalman)


    # Frequency demodulation demo
    # d = 2
    # hdyn = {'alpha': 10.0, 'el': 30.0 * np.ones(d, )}
    # hobs = {'alpha': 10.0, 'el': 30.0 * np.ones(d, )}
    # from models.demodulation import frequency_demodulation_filter_demo
    # frequency_demodulation_filter_demo(GPQKalman, 'rbf', 'sr', kern_hyp_dyn=hdyn, kern_hyp_obs=hobs)
    # frequency_demodulation_filter_demo(GPQMKalman, 'rbf', 'sr')


if __name__ == '__main__':
    main()
