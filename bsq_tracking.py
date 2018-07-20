import numpy as np
from numpy import newaxis as na
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import OrderedDict
import time

from ssmtoybox.ssinf import GaussianProcessKalman, BayesSardKalman, UnscentedKalman
from ssmtoybox.ssmod import ReentryVehicleRadarTrackingSimpleGaussSSM, ReentryVehicleRadarTrackingGaussSSM
from ssmtoybox.dynsys import ReentryVehicleRadarTrackingSimpleGaussSystem, ReentryVehicleRadarTrackingGaussSystem
from ssmtoybox.utils import mse_matrix, squared_error, log_cred_ratio


np.set_printoptions(precision=4)


def reentry_simple_demo(dur=30, tau=0.1, mc=100):
    """
    A spherical object falls down from high altitude entering the Earth’s atmosphere with a high velocity.

    Parameters
    ----------
    tau: float
        discretization period for the dynamics ODE integration method
    dur: int
        Duration of the dynamics simulation
    mc: int
        Number of Monte Carlo simulations.

    Notes
    -----
    The parameter mc determines the number of trajectories simulated.

    Returns
    -------
    """

    # Generate reference trajectory by ODE integration
    sys = ReentryVehicleRadarTrackingSimpleGaussSystem()
    x = sys.simulate_trajectory(method='rk4', dt=tau, duration=dur, mc_sims=mc)

    # pick only non-divergent trajectories
    x = x[..., np.all(x >= 0, axis=(0, 1))]
    mc = x.shape[2]

    y = np.zeros((sys.zD,) + x.shape[1:])
    for i in range(mc):
        y[..., i] = sys.simulate_measurements(x[..., i], mc_per_step=1).squeeze()

    # Initialize model
    ssm = ReentryVehicleRadarTrackingSimpleGaussSSM(dt=tau)

    # GPQKF kernel parameters
    kpar_dyn_ut = np.array([[0.5, 10, 10, 10]])
    kpar_obs_ut = np.array([[0.5, 15, 20, 20]])

    # UT multi-index
    mul_ut = np.hstack((np.zeros((ssm.xD, 1)), np.eye(ssm.xD), 2*np.eye(ssm.xD))).astype(np.int)

    # Initialize filters
    alg = (
        GaussianProcessKalman(ssm, kpar_dyn_ut, kpar_obs_ut, kernel='rbf', points='ut'),
        BayesSardKalman(ssm, kpar_dyn_ut, kpar_obs_ut, mul_ut, mul_ut, points='ut'),
        UnscentedKalman(ssm, beta=0),
    )

    num_alg = len(alg)
    d, steps, mc = x.shape
    mean, cov = np.zeros((d, steps, mc, num_alg)), np.zeros((d, d, steps, mc, num_alg))
    for imc in range(mc):
        for ia, a in enumerate(alg):
            # Do filtering and reset the filters for each new track
            mean[..., imc, ia], cov[..., imc, ia] = a.forward_pass(y[..., imc])
            a.reset()

    # time index for plotting
    time_ind = np.linspace(1, dur, x.shape[1])

    # PLOTS: Trajectories
    # plt.figure()
    # g = GridSpec(4, 2)
    # plt.subplot(g[:2, :])
    #
    # # Earth surface w/ radar position
    # t = np.arange(0.48 * np.pi, 0.52 * np.pi, 0.01)
    # plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t) - sys.R0, 'darkblue', lw=2)
    # plt.plot(sys.sx, sys.sy, 'ko')
    #
    # xzer = np.zeros(x.shape[1])
    # for i in range(mc):
    #     # Vehicle trajectory
    #     plt.plot(xzer, x[0, :, i], alpha=0.35, color='r', ls='--', lw=2)
    #
    #     # Filtered position estimate
    #     plt.plot(xzer, mean[0, :, i, 0], color='g', alpha=0.3)
    #     plt.plot(xzer, mean[0, :, i, 1], color='orange', alpha=0.3)

    # Altitude
    # x0 = sys.pars['x0_mean']
    # plt.subplot(g[2, :])
    # plt.ylim([0, x0[0]])
    # for i in range(mc):
    #     plt.plot(time_ind, x[0, :, i], alpha=0.35, color='b')
    # plt.ylabel('altitude [ft]')
    # plt.xlabel('time [s]')
    #
    # # Velocity
    # plt.subplot(g[3, :])
    # plt.ylim([0, x0[1]])
    # for i in range(mc):
    #     plt.plot(time_ind, x[1, :, i], alpha=0.35, color='b')
    # plt.ylabel('velocity [ft/s]')
    # plt.xlabel('time [s]')

    # Compute Performance Scores
    error2 = mean.copy()
    pos_lcr = np.zeros((steps, mc, num_alg))
    vel_lcr = pos_lcr.copy()
    theta_lcr = pos_lcr.copy()
    for a in range(num_alg):
        for k in range(steps):
            pos_mse = mse_matrix(x[0, na, k, :], mean[0, na, k, :, a])
            vel_mse = mse_matrix(x[1, na, k, :], mean[1, na, k, :, a])
            theta_mse = mse_matrix(x[2, na, k, :], mean[2, na, k, :, a])
            for imc in range(mc):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                pos_lcr[k, imc, a] = log_cred_ratio(x[0, k, imc], mean[0, k, imc, a],
                                                    cov[0, 0, k, imc, a], pos_mse)
                vel_lcr[k, imc, a] = log_cred_ratio(x[1, k, imc], mean[1, k, imc, a],
                                                    cov[1, 1, k, imc, a], vel_mse)
                theta_lcr[k, imc, a] = log_cred_ratio(x[2, k, imc], mean[2, k, imc, a],
                                                      cov[2, 2, k, imc, a], theta_mse)

    # Averaged position/velocity RMSE and inclination in time
    pos_rmse = np.sqrt(error2[0, na, ...].sum(axis=0))
    pos_rmse_vs_time = pos_rmse.mean(axis=1)
    pos_inc_vs_time = pos_lcr.mean(axis=1)
    vel_rmse = np.sqrt(error2[1, na, ...].sum(axis=0))
    vel_rmse_vs_time = vel_rmse.mean(axis=1)
    vel_inc_vs_time = vel_lcr.mean(axis=1)
    theta_rmse = np.sqrt(error2[2, na, ...].sum(axis=0))
    theta_rmse_vs_time = theta_rmse.mean(axis=1)
    theta_inc_vs_time = theta_lcr.mean(axis=1)

    # PLOTS: RMSE in time for position, velocity and ballistic parameter
    plt.figure()
    g = GridSpec(6, 3)
    # filt_labels = ['UT', 'CUT', 'GPQ-UT', 'GPQ-CUT']
    filt_labels = ['GPQKF', 'BSQKF', 'UKF']
    plt.subplot(g[:2, :2])
    plt.ylabel('RMSE')
    for i in range(num_alg):
        plt.plot(time_ind[1:], pos_rmse_vs_time[1:, i], label=filt_labels[i])
    plt.legend()

    plt.subplot(g[2:4, :2])
    plt.ylabel('RMSE')
    for i in range(num_alg):
        plt.plot(time_ind[1:], vel_rmse_vs_time[1:, i], label=filt_labels[i])

    plt.subplot(g[4:, :2])
    plt.ylabel('RMSE')
    plt.xlabel('time step [k]')
    for i in range(num_alg):
        plt.plot(time_ind[1:], theta_rmse_vs_time[1:, i], label=filt_labels[i])

    # BOX PLOTS: time-averaged RMSE
    plt.subplot(g[:2, 2:])
    plt.boxplot(pos_rmse.mean(axis=0), labels=filt_labels)

    plt.subplot(g[2:4, 2:])
    plt.boxplot(vel_rmse.mean(axis=0), labels=filt_labels)

    plt.subplot(g[4:, 2:])
    plt.boxplot(theta_rmse.mean(axis=0), labels=filt_labels)
    plt.show()

    # PLOTS: Inclination indicator in time for position, velocity and ballistic parameter
    plt.figure()
    g = GridSpec(6, 3)

    plt.subplot(g[:2, :2])
    plt.ylabel(r'$\nu$')
    for i in range(num_alg):
        plt.plot(time_ind, pos_inc_vs_time[:, i], label=filt_labels[i])

    plt.subplot(g[2:4, :2])
    plt.ylabel(r'$\nu$')
    for i in range(num_alg):
        plt.plot(time_ind, vel_inc_vs_time[:, i], label=filt_labels[i])

    plt.subplot(g[4:, :2])
    plt.ylabel(r'$\nu$')
    plt.xlabel('time step [k]')
    for i in range(num_alg):
        plt.plot(time_ind, theta_inc_vs_time[:, i], label=filt_labels[i])

    # BOX PLOTS: time-averaged inclination indicator
    plt.subplot(g[:2, 2:])
    plt.boxplot(pos_lcr.mean(axis=0), labels=filt_labels)

    plt.subplot(g[2:4, 2:])
    plt.boxplot(vel_lcr.mean(axis=0), labels=filt_labels)

    plt.subplot(g[4:, 2:])
    plt.boxplot(theta_lcr.mean(axis=0), labels=filt_labels)
    plt.show()

    # TODO: pandas tables for printing into latex
    np.set_printoptions(precision=4)
    print('Average RMSE: {}'.format(np.sqrt(error2.sum(axis=0)).mean(axis=(0, 1))))


def reentry_demo():
    mc_sims = 20
    disc_tau = 0.5  # discretization period

    # Generate reference trajectory by ODE integration
    sys = ReentryVehicleRadarTrackingGaussSystem()
    x = sys.simulate_trajectory(method='rk4', dt=disc_tau, duration=200, mc_sims=mc_sims)
    y = np.zeros((sys.zD,) + x.shape[1:])
    for i in range(mc_sims):
        y[..., i] = sys.simulate_measurements(x[..., i], mc_per_step=1).squeeze()

    # Initialize state-space model with mis-specified initial mean
    ssm = ReentryVehicleRadarTrackingGaussSSM(dt=disc_tau)
    ssm.set_pars('x0_mean', np.array([6500.4, 349.14, -1.1093, -6.1967, 0.6932]))

    # Initialize filters
    par_dyn = np.array([[1.0, 1, 1, 1, 1, 1]])
    par_obs = np.array([[1.0, 0.9, 0.9, 1e4, 1e4, 1e4]])  # np.atleast_2d(np.ones(6))
    mul_ut = np.hstack((np.zeros((ssm.xD, 1)), np.eye(ssm.xD), 2 * np.eye(ssm.xD))).astype(np.int)
    alg = OrderedDict({
        # GaussianProcessKalman(ssm, par_dyn, par_obs, kernel='rbf', points='ut'),
        'bsqkf': BayesSardKalman(ssm, par_dyn, par_obs, mul_ut, mul_ut, points='ut'),
        'ukf': UnscentedKalman(ssm, beta=0),
    })

    kpdyn = np.array([[0.5, 1, 1e3, 1, 1e3, 1e3],
                      [0.5, 1e3, 1, 1e3, 1, 1e3],
                      [0.35, 3, 3, 3, 3, 1],
                      [0.35, 3, 3, 3, 3, 1],
                      [2.2, 1e3, 1e3, 1e3, 1e3, 1]])
    alg['bsqkf'].tf_dyn.model.model_var = multivariate_emv(alg['bsqkf'].tf_dyn, kpdyn, mul_ut)
    kpobs = np.array([[1.0, 1, 1, 1e2, 1e2, 1e2],
                      [1.0, 1.4, 1.4, 1e2, 1e2, 1e2]])
    # multivariate_emv(alg[0].tf_meas, kpobs, mul_ut)  # 1e-8*np.eye(2)
    alg['bsqkf'].tf_meas.model.model_var = 0*np.eye(2)
    print('BSQ EMV\ndyn: {} \nobs: {}'.format(alg['bsqkf'].tf_dyn.model.model_var.diagonal(),
                                              alg['bsqkf'].tf_meas.model.model_var.diagonal()))

    # Are both filters using the same sigma-points?
    # assert np.array_equal(alg[0].tf_dyn.model.points, alg[1].tf_dyn.unit_sp)

    num_alg = len(alg)
    d, steps, mc_sims = x.shape
    mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
    for ia, a in enumerate(alg):
        print('Running {:<5} ... '.format(a.upper()), end='', flush=True)
        t0 = time.time()
        for imc in range(mc_sims):
            mean[..., imc, ia], cov[..., imc, ia] = alg[a].forward_pass(y[..., imc])
            alg[a].reset()
        print('{:>30}'.format('Done in {:.2f} [sec]'.format(time.time() - t0)))

    # Performance score plots
    print()
    print('Computing performance scores ...')
    error2 = mean.copy()
    lcr_pos = np.zeros((steps, mc_sims, num_alg))
    lcr_vel = lcr_pos.copy()
    lcr_theta = lcr_pos.copy()
    for a in range(num_alg):
        for k in range(steps):
            mse = mse_matrix(x[:, k, :], mean[:, k, :, a])
            for imc in range(mc_sims):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                lcr_pos[k, imc, a] = log_cred_ratio(x[:2, k, imc], mean[:2, k, imc, a],
                                                    cov[:2, :2, k, imc, a], mse[:2, :2])
                lcr_vel[k, imc, a] = log_cred_ratio(x[2:4, k, imc], mean[2:4, k, imc, a],
                                                    cov[2:4, 2:4, k, imc, a], mse[2:4, 2:4])
                lcr_theta[k, imc, a] = log_cred_ratio(x[4, k, imc], mean[4, k, imc, a],
                                                      cov[4, 4, k, imc, a], mse[4, 4])

    # Averaged RMSE and Inclination Indicator in time
    pos_rmse_vs_time = np.sqrt((error2[:2, ...]).sum(axis=0)).mean(axis=1)
    vel_rmse_vs_time = np.sqrt((error2[2:4, ...]).sum(axis=0)).mean(axis=1)
    theta_rmse_vs_time = np.sqrt((error2[4, ...])).mean(axis=1)
    pos_inc_vs_time = lcr_pos.mean(axis=1)
    vel_inc_vs_time = lcr_vel.mean(axis=1)
    theta_inc_vs_time = lcr_theta.mean(axis=1)

    # print performance scores
    print('Average position RMSE: {}'.format(pos_rmse_vs_time.mean(axis=0)))
    print('Average position Inc.: {}'.format(pos_inc_vs_time.mean(axis=0)))
    print('Average velocity RMSE: {}'.format(vel_rmse_vs_time.mean(axis=0)))
    print('Average velocity Inc.: {}'.format(vel_inc_vs_time.mean(axis=0)))
    print('Average parameter RMSE: {}'.format(theta_rmse_vs_time.mean(axis=0)))
    print('Average parameter Inc.: {}'.format(theta_inc_vs_time.mean(axis=0)))

    # PLOTS # TODO: figures for states rather than scores
    # root mean squared error
    plt.figure().suptitle('RMSE')
    g = GridSpec(3, 1)
    ax = plt.subplot(g[0, 0])
    ax.set_title('Position')
    for i, f_str in enumerate(alg):
        ax.plot(pos_rmse_vs_time[:, i], label=f_str.upper())
    ax.legend()
    ax = plt.subplot(g[1, 0])
    ax.set_title('Velocity')
    for i, f_str in enumerate(alg):
        ax.plot(vel_rmse_vs_time[:, i])
    ax = plt.subplot(g[2, 0])
    ax.set_title('Parameter')
    for i, f_str in enumerate(alg):
        ax.plot(theta_rmse_vs_time[:, i])
    plt.show(block=False)

    # inclination
    plt.figure().suptitle('Inclination')
    ax = plt.subplot(g[0, 0])
    ax.set_title('Position')
    for i, f_str in enumerate(alg):
        ax.plot(pos_inc_vs_time[:, i], label=f_str.upper())
    ax.legend()
    ax = plt.subplot(g[1, 0])
    ax.set_title('Velocity')
    for i, f_str in enumerate(alg):
        ax.plot(vel_inc_vs_time[:, i])
    ax = plt.subplot(g[2, 0])
    ax.set_title('Parameter')
    for i, f_str in enumerate(alg):
        ax.plot(theta_inc_vs_time[:, i])
    plt.show()


def multivariate_emv(tf, par, multi_ind):
    import scipy.linalg as la
    from ssmtoybox.utils import vandermonde
    dim, num_basis = multi_ind.shape
    num_out, num_par = par.shape
    x = tf.model.points
    emv = np.eye(num_out)
    for i in range(num_out):
        # inverse kernel matrix and Vandermonde matrix
        iK = tf.model.kernel.eval_inv_dot(par[i, :], x, scaling=False)
        V = vandermonde(multi_ind, x)
        iV = la.solve(V, np.eye(num_basis))
        iViKV = la.cho_solve(la.cho_factor(V.T.dot(iK).dot(V) + 1e-8 * np.eye(num_basis)), np.eye(num_basis))
        # expectations of multivariate polynomials
        pxpx = tf.model._exp_x_pxpx(multi_ind)
        kxpx = tf.model._exp_x_kxpx(par[i, :], multi_ind, x)
        # kernel expectations
        emv[i, i] = par[i, 0] * (1 - np.trace(kxpx.T.dot(iV.T) + kxpx.dot(iV) - pxpx.dot(iViKV)))
    return emv


if __name__ == '__main__':
    reentry_demo()
