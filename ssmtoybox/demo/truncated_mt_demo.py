import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ssmtoybox.mtran import MomentTransform
from ssmtoybox.mtran import MonteCarloTransform, UnscentedTransform, TruncatedUnscentedTransform
from ssmtoybox.ssinf import TruncatedUnscentedKalman, UnscentedKalman
from ssmtoybox.ssmod import ReentryVehicle2DTransition, Radar2DMeasurement
from ssmtoybox.utils import GaussRV
from ssmtoybox.utils import ellipse_points, symmetrized_kl_divergence, squared_error, log_cred_ratio, mse_matrix


def cartesian2polar(x, pars, dx=False):
    return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan2(x[1], x[0])])


def polar2cartesian(x, pars, dx=False):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def mt_trunc_demo(mt_trunc, mt, dim=None, full_input_cov=True, **kwargs):
    """
    Comparison of truncated MT and vanilla MT on polar2cartesian transform for increasing state dimensions. The
    truncated MT is aware of the effective dimension, so we expect it to be closer to the true covariance

    Observation: Output covariance of the Truncated UT stays closer to the MC baseline than the covariance
    produced by the vanilla UT.

    There's some merit to the idea, but the problem is with computing the input-output cross-covariance.


    Parameters
    ----------
    mt_trunc
    mt
    dim
    full_input_cov : boolean
        If `False`, a diagonal input covariance is used, otherwise a full covariance is used.
    kwargs

    Returns
    -------

    """

    assert issubclass(mt_trunc, MomentTransform) and issubclass(mt, MomentTransform)

    # state dimensions and effective dimension
    dim = [2, 3, 4, 5] if dim is None else dim
    d_eff = 2

    # nonlinear integrand
    f = polar2cartesian

    # input mean and covariance
    mean_eff, cov_eff = np.array([1, np.pi / 2]), np.diag([0.05 ** 2, (np.pi / 10) ** 2])

    if full_input_cov:
        A = np.random.rand(d_eff, d_eff)
        cov_eff = A.dot(cov_eff).dot(A.T)

    # use MC transform with lots of samples to compute the true transformed moments
    tmc = MonteCarloTransform(d_eff, n=1e4)
    M_mc, C_mc, cc_mc = tmc.apply(f, mean_eff, cov_eff, None)
    # transformed samples
    x = np.random.multivariate_normal(mean_eff, cov_eff, size=int(1e3)).T
    fx = np.apply_along_axis(f, 0, x, None)
    X_mc = ellipse_points(M_mc, C_mc)

    M = np.zeros((2, len(dim), 2))
    C = np.zeros((2, 2, len(dim), 2))
    X = np.zeros((2, 50, len(dim), 2))
    for i, d in enumerate(dim):
        t = mt_trunc(d, d_eff, **kwargs)
        s = mt(d, **kwargs)

        # input mean and covariance
        mean, cov = np.zeros(d), np.eye(d)
        mean[:d_eff], cov[:d_eff, :d_eff] = mean_eff, cov_eff

        # transformed moments (w/o cross-covariance)
        M[:, i, 0], C[..., i, 0], cc = t.apply(f, mean, cov, None)
        M[:, i, 1], C[..., i, 1], cc = s.apply(f, mean, cov, None)

        # points on the ellipse defined by the transformed mean and covariance for plotting
        X[..., i, 0] = ellipse_points(M[:, i, 0], C[..., i, 0])
        X[..., i, 1] = ellipse_points(M[:, i, 1], C[..., i, 1])

    # PLOTS: transformed samples, MC mean and covariance ground truth
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(fx[0, :], fx[1, :], 'k.', alpha=0.15)
    ax[0].plot(M_mc[0], M_mc[1], 'ro', markersize=6, lw=2)
    ax[0].plot(X_mc[0, :], X_mc[1, :], 'r--', lw=2, label='MC')

    # SR and SR-T mean and covariance for various state dimensions
    # TODO: it's more effective to plot SKL between the transformed and baseline covariances.
    for i, d in enumerate(dim):
        ax[0].plot(M[0, i, 0], M[1, i, 0], 'b+', markersize=10, lw=2)
        ax[0].plot(X[0, :, i, 0], X[1, :, i, 0], color='b', label='mt-trunc (d={})'.format(d))
    for i, d in enumerate(dim):
        ax[0].plot(M[0, i, 1], M[1, i, 1], 'go', markersize=6)
        ax[0].plot(X[0, :, i, 1], X[1, :, i, 1], color='g', label='mt (d={})'.format(d))
    ax[0].set_aspect('equal')
    plt.legend()

    # symmetrized KL-divergence
    skl = np.zeros((len(dim), 2))
    for i, d in enumerate(dim):
        skl[i, 0] = symmetrized_kl_divergence(M_mc, C_mc, M[:, i, 0], C[..., i, 0])
        skl[i, 1] = symmetrized_kl_divergence(M_mc, C_mc, M[:, i, 1], C[..., i, 1])
    plt_opt = {'lw': 2, 'marker': 'o'}
    ax[1].plot(dim, skl[:, 0], label='truncated', **plt_opt)
    ax[1].plot(dim, skl[:, 1], label='original', **plt_opt)
    ax[1].set_xticks(dim)
    ax[1].set_xlabel('Dimension')
    ax[1].set_ylabel('SKL')
    plt.legend()
    plt.show()


def ukf_trunc_demo(mc_sims=50):
    disc_tau = 0.5  # discretization period in seconds
    duration = 200

    # define system
    m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932])
    P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 0])
    x0 = GaussRV(5, m0, P0)
    q = GaussRV(3, cov=np.diag([2.4064e-5, 2.4064e-5, 0]))
    sys = ReentryVehicle2DTransition(x0, q, dt=disc_tau)
    # define radar measurement model
    r = GaussRV(2, cov=np.diag([1e-6, 0.17e-6]))
    obs = Radar2DMeasurement(r, sys.dim_state)

    # simulate reference state trajectory by SDE integration
    x = sys.simulate_continuous(duration, disc_tau, mc_sims)
    x_ref = x.mean(axis=2)
    # simulate corresponding radar measurements
    y = obs.simulate_measurements(x)

    # initialize state-space model; uses cartesian2polar as measurement (not polar2cartesian)
    P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
    x0 = GaussRV(5, m0, P0)
    q = GaussRV(3, cov=np.diag([2.4064e-5, 2.4064e-5, 1e-6]))
    dyn = ReentryVehicle2DTransition(x0, q, dt=disc_tau)

    # initialize UKF and UKF in truncated version
    alg = (
        UnscentedKalman(dyn, obs),
        TruncatedUnscentedKalman(dyn, obs),
    )
    num_alg = len(alg)

    # space for filtered mean and covariance
    steps = x.shape[1]
    x_mean = np.zeros((dyn.dim_in, steps, mc_sims, num_alg))
    x_cov = np.zeros((dyn.dim_in, dyn.dim_in, steps, mc_sims, num_alg))

    # filtering estimate of the state trajectory based on provided measurements
    from tqdm import trange
    for i_est, estimator in enumerate(alg):
        for i_mc in trange(mc_sims):
            x_mean[..., i_mc, i_est], x_cov[..., i_mc, i_est] = estimator.forward_pass(y[..., i_mc])
            estimator.reset()

    # Plots
    plt.figure()
    g = GridSpec(2, 4)
    plt.subplot(g[:, :2])

    # Earth surface w/ radar position
    radar_x, radar_y = dyn.R0, 0
    t = 0.02 * np.arange(-1, 4, 0.1)
    plt.plot(dyn.R0 * np.cos(t), dyn.R0 * np.sin(t), color='darkblue', lw=2)
    plt.plot(radar_x, radar_y, 'ko')

    plt.plot(x_ref[0, :], x_ref[1, :], color='r', ls='--')
    # Convert from polar to cartesian
    meas = np.stack(( + y[0, ...] * np.cos(y[1, ...]), radar_y + y[0, ...] * np.sin(y[1, ...])), axis=0)
    for i in range(mc_sims):
        # Vehicle trajectory
        # plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='r', ls='--')

        # Plot measurements
        plt.plot(meas[0, :, i], meas[1, :, i], 'k.', alpha=0.3)

        # Filtered position estimate
        plt.plot(x_mean[0, 1:, i, 0], x_mean[1, 1:, i, 0], color='g', alpha=0.3)
        plt.plot(x_mean[0, 1:, i, 1], x_mean[1, 1:, i, 1], color='orange', alpha=0.3)

    # Performance score plots
    error2 = x_mean.copy()
    lcr = np.zeros((steps, mc_sims, num_alg))
    for a in range(num_alg):
        for k in range(steps):
            mse = mse_matrix(x[:4, k, :], x_mean[:4, k, :, a])
            for imc in range(mc_sims):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], x_mean[:, k, imc, a])
                lcr[k, imc, a] = log_cred_ratio(x[:4, k, imc], x_mean[:4, k, imc, a], x_cov[:4, :4, k, imc, a], mse)

    # Averaged RMSE and Inclination Indicator in time
    pos_rmse_vs_time = np.sqrt((error2[:2, ...]).sum(axis=0)).mean(axis=1)
    inc_ind_vs_time = lcr.mean(axis=1)

    # Plots
    plt.subplot(g[0, 2:])
    plt.title('RMSE')
    plt.plot(pos_rmse_vs_time[:, 0], label='UKF', color='g')
    plt.plot(pos_rmse_vs_time[:, 1], label='UKF-trunc', color='r')
    plt.legend()
    plt.subplot(g[1, 2:])
    plt.title('Inclination Indicator $I^2$')
    plt.plot(inc_ind_vs_time[:, 0], label='UKF', color='g')
    plt.plot(inc_ind_vs_time[:, 1], label='UKF-trunc', color='r')
    plt.legend()
    plt.show()

    print('Average RMSE: {}'.format(pos_rmse_vs_time.mean(axis=0)))
    print('Average I2: {}'.format(inc_ind_vs_time.mean(axis=0)))


if __name__ == '__main__':
    # truncated transform significantly improves the SKL between the true and approximate Gaussian
    mt_trunc_demo(TruncatedUnscentedTransform, UnscentedTransform, [2, 5, 10, 15], kappa=0, full_input_cov=True)

    # truncated transform performance virtually identical when applied in filtering
    # ukf_trunc_demo(mc_sims=50)
