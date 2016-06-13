from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import GPQ


class GPQKalman(StateSpaceInference):
    """
    GP quadrature filter and smoother.
    """

    def __init__(self, sys, kernel, points, kern_hyp_dyn=None, kern_hyp_obs=None, point_hyp=None):
        assert isinstance(sys, StateSpaceModel)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        t_dyn = GPQ(nq, kernel, points, kern_hyp_dyn, point_hyp)
        t_obs = GPQ(nr, kernel, points, kern_hyp_obs, point_hyp)
        super(GPQKalman, self).__init__(sys, t_dyn, t_obs)


def main():
    # UNGM demo
    # from models.ungm import ungm_filter_demo
    # khyp = {'alpha': 1.0, 'el': 0.3 * np.ones(1)}
    # ut_hyp = {'kappa': 0.0}
    # ungm_filter_demo(GPQKalman, 'rbf', 'sr', kern_hyp_dyn=khyp, kern_hyp_obs=khyp, point_hyp=None)

    # Pendulum demo
    # from models.pendulum import pendulum_filter_demo
    # hdyn, hmeas = None, None
    # pendulum_filter_demo(GPQKalman, hyp_dyn=hdyn, hyp_meas=hmeas)

    # Reentry vehicle tracking demo
    # The radar measurement model only uses the first two state dimensions as input, which means that the remaining
    # state dimensions are irrelevant to the GP model of the measurement function. This is why we set high
    # lengthscale for the remaining dimensions, which ensures they will not contribute significantly to kernel
    # covariance (the RBF kernel is expressed in terms of inverse lengthscales).
    # TODO: find hypers that give position RMSE < ~0.01 (UKF), GPQKF best position RMSE is ~0.1,
    from models.tracking import reentry_filter_demo
    d = 5
    hdyn = {'alpha': 1.0, 'el': [15.0, 15.0, 15.0, 15.0, 15.0]}
    hobs = {'alpha': 1.0, 'el': [15.0, 15.0, 1e4, 1e4, 1e4]}
    reentry_filter_demo(GPQKalman, 'rbf', 'ut', kern_hyp_dyn=hdyn, kern_hyp_obs=hobs)

    # Frequency demodulation demo
    # d = 2
    # hdyn = {'alpha': 10.0, 'el': 30.0 * np.ones(d, )}
    # hobs = {'alpha': 10.0, 'el': 30.0 * np.ones(d, )}
    # from models.demodulation import frequency_demodulation_filter_demo
    # frequency_demodulation_filter_demo(GPQKalman, 'rbf', 'sr', kern_hyp_dyn=hdyn, kern_hyp_obs=hobs)


if __name__ == '__main__':
    main()
