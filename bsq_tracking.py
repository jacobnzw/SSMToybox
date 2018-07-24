import numpy as np
import pandas as pd
from numpy import newaxis as na
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import OrderedDict
import time

from ssmtoybox.ssinf import GaussianProcessKalman, BayesSardKalman, UnscentedKalman, ExtendedKalman, GaussianInference
from ssmtoybox.ssmod import ReentryVehicleRadarTrackingSimpleGaussSSM, ReentryVehicleRadarTrackingGaussSSM, \
    CoordinatedTurnBearingsOnlyTrackingGaussSSM
from ssmtoybox.dynsys import ReentryVehicleRadarTrackingSimpleGaussSystem, ReentryVehicleRadarTrackingGaussSystem
from ssmtoybox.utils import mse_matrix, squared_error, log_cred_ratio
from ssmtoybox.ssmod import GaussianStateSpaceModel, StateSpaceModel
from ssmtoybox.bq.bqmtran import BayesSardTransform
from ssmtoybox.mtran import LinearizationTransform


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


def coordinated_turn_radar_demo():
    num_steps = 100
    mc_sims = 20

    # sensor positions
    x_min, x_max = -10000, 10000
    y_min, y_max = -10000, 10000
    S = np.array([[x_min, y_min],
                  [x_min, y_max],
                  [x_max, y_min],
                  [x_max, y_max]])
    tau = 1.0
    # generate data
    sys = CoordinatedTurnBearingsOnlyTrackingGaussSSM(dt=tau, sensor_pos=S)
    x, y = sys.simulate(num_steps, mc_sims)

    # weed out trajectories venturing outside of the sensor rectangle
    ix = np.all(np.abs(x[(0, 2), ...]) <= x_max, axis=(0, 1))
    x, y = x[..., ix], y[..., ix]
    print('{:.2f}% of trajectories weeded out.'.format(100 * np.count_nonzero(ix == False) / len(ix)))

    # state-space model for filters
    ssm = CoordinatedTurnBearingsOnlyTrackingGaussSSM(dt=tau, sensor_pos=S)
    # Initialize filters
    par_dyn = np.array([[1.0, 1, 1, 1, 1, 1]])
    par_obs = np.array([[1.0, 0.9, 0.9, 1e4, 1e4, 1e4]])
    mul_ut = np.hstack((np.zeros((ssm.xD, 1)), np.eye(ssm.xD), 2 * np.eye(ssm.xD))).astype(np.int)
    alg = OrderedDict({
        # GaussianProcessKalman(ssm, par_dyn, par_obs, kernel='rbf', points='ut'),
        'bsqkf': BayesSardKalman(ssm, par_dyn, par_obs, mul_ut, mul_ut, points='ut'),
        'ukf': UnscentedKalman(ssm, beta=0),
    })

    kpdyn = np.array([[1.0, 1, 1e3, 1, 1e3, 1e3],
                      [1.0, 1e3, 1, 1e3, 1, 1e3],
                      [1.0, 3, 3, 3, 3, 1],
                      [1.0, 3, 3, 3, 3, 1],
                      [1.0, 1e3, 1e3, 1e3, 1e3, 1]])
    alg['bsqkf'].tf_dyn.model.model_var = 1e-3*np.eye(5)  # multivariate_emv(alg['bsqkf'].tf_dyn, kpdyn, mul_ut)
    kpobs = np.array([[1.0, 1, 1, 1e2, 1e2, 1e2],
                      [1.0, 1.4, 1.4, 1e2, 1e2, 1e2]])
    # multivariate_emv(alg[0].tf_meas, kpobs, mul_ut)  # 1e-8*np.eye(2)
    alg['bsqkf'].tf_meas.model.model_var = 1e-3*np.eye(4)
    print('BSQ EMV\ndyn: {} \nobs: {}'.format(alg['bsqkf'].tf_dyn.model.model_var.diagonal(),
                                              alg['bsqkf'].tf_meas.model.model_var.diagonal()))

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
                cov_sel = cov[..., k, imc, a]
                lcr_pos[k, imc, a] = log_cred_ratio(x[(0, 2), k, imc], mean[(0, 2), k, imc, a],
                                                    cov_sel[np.ix_((0, 2), (0, 2))], mse[np.ix_((0, 2), (0, 2))])
                lcr_vel[k, imc, a] = log_cred_ratio(x[(1, 3), k, imc], mean[(1, 3), k, imc, a],
                                                    cov_sel[np.ix_((1, 3), (1, 3))], mse[np.ix_((1, 3), (1, 3))])
                lcr_theta[k, imc, a] = log_cred_ratio(x[4, k, imc], mean[4, k, imc, a],
                                                      cov[4, 4, k, imc, a], mse[4, 4])

    # Averaged RMSE and Inclination Indicator in time
    pos_rmse_vs_time = np.sqrt((error2[(0, 2), ...]).sum(axis=0)).mean(axis=1)
    vel_rmse_vs_time = np.sqrt((error2[(1, 3), ...]).sum(axis=0)).mean(axis=1)
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

    # set model variance
    tf.model.model_var = emv
    return emv


class ConstantVelocityRadarTrackingGaussSSM(GaussianStateSpaceModel):
    xD = 4
    zD = 2
    qD = 2
    rD = 2

    q_additive = True
    r_additive = True

    def __init__(self, dt=0.5):
        self.dt = dt
        self.q_gain = np.array([[dt ** 2 / 2, 0],
                                [dt, 0],
                                [0, dt ** 2 / 2],
                                [0, dt]])
        pars = {
            'x0_mean': np.array([10175, 295, 980, -35]),  # m, m/s, m, m/s
            'x0_cov': np.diag([100**2, 10**2, 100**2, 10**2]),
            'q_mean': np.zeros((self.qD, )),
            'q_cov': np.diag([50, 5]),  # m^2/s^4, m^2/s^4
            'q_gain': self.q_gain,
            'r_mean': np.zeros((self.rD, )),
            # 'r_cov': np.diag([50, 0.4]),  # m^2, mrad^2
            'r_cov': np.diag([50, 0.4e-6]),  # m^2, rad^2
        }
        super(ConstantVelocityRadarTrackingGaussSSM, self).__init__(**pars)

    def dyn_fcn(self, x, q, pars):
        A = np.array([[1, self.dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, self.dt],
                      [0, 0, 0, 1]])
        return A.dot(x) + self.q_gain.dot(q)

    def meas_fcn(self, x, r, pars):
        rang = np.sqrt(x[0]**2 + x[2]**2)
        theta = np.arctan2(x[2], x[0])
        return np.array([rang, theta]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        return np.array([[1, self.dt, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, self.dt],
                         [0, 0, 0, 1]]).T

    def meas_fcn_dx(self, x, r, pars):
        rang = np.sqrt(x[0] ** 2 + x[2] ** 2)
        return np.array([[x[0]/rang, 0, x[2]/rang, 0],
                         [-x[2]/(rang**2), 0, x[0]/(rang**2), 0]])


class LinearBayesSardKalman(GaussianInference):
    """
    Gaussian filter with linear MT on the dynamics model and BSQ-MT on observation model.
    """
    def __init__(self, ssm, kern_par_obs, mulind_obs=2, points='ut', point_hyp=None):
        assert isinstance(ssm, StateSpaceModel)
        nq = ssm.xD if ssm.q_additive else ssm.xD + ssm.qD
        nr = ssm.xD if ssm.r_additive else ssm.xD + ssm.rD
        t_dyn = LinearizationTransform(nq)
        t_obs = BayesSardTransform(nr, ssm.zD, kern_par_obs, mulind_obs, points, point_hyp)
        super(LinearBayesSardKalman, self).__init__(ssm, t_dyn, t_obs)


def constant_velocity_radar_tracking_demo(steps=100, mc_sims=100):

    sys = ConstantVelocityRadarTrackingGaussSSM()
    sys.set_pars('x0_mean', np.array([10000, 300, 1000, -40]))
    x, y = sys.simulate(steps, mc_sims)

    sys.check_jacobians(h=np.sqrt(np.finfo(float).eps))

    # state-space model for filters
    ssm = ConstantVelocityRadarTrackingGaussSSM()
    # ssm.set_pars('x0_mean', np.array([10000, 300, 1000, -40]))
    # kernel parameters for dynamics and observation functions
    kpar_dyn = np.array([[1.0] + ssm.xD * [10]])
    kpar_obs = np.array([[1.0] + ssm.xD * [1]])
    alpha_ut = np.hstack((np.zeros((ssm.xD, 1)), np.eye(ssm.xD), 2*np.eye(ssm.xD))).astype(np.int)
    alg = OrderedDict({
        'bsqkf': BayesSardKalman(ssm, kpar_dyn, kpar_obs, alpha_ut, alpha_ut, points='ut'),
        'lbsqkf': LinearBayesSardKalman(ssm, kpar_obs, alpha_ut, points='ut'),
        'ukf': UnscentedKalman(ssm),
        # 'ekf': ExtendedKalman(ssm),
    })

    # set custom model variance
    kpobs = np.array([[0.00001, 1, 2, 1e2, 1e2],
                      [0.00001, 1, 2, 1e2, 1e2]])
    # multivariate_emv(alg['bsqkf'].tf_meas, kpobs, alpha_ut)  # 1e-3 * np.eye(4)
    # multivariate_emv(alg['lbsqkf'].tf_meas, kpobs, alpha_ut)
    alg['bsqkf'].tf_meas.model.model_var = 1e-8 * np.eye(2)
    alg['lbsqkf'].tf_meas.model.model_var = 1e-8 * np.eye(2)

    print('BSQKF Expected Model Variance')
    print('DYN: {}'.format(alg['bsqkf'].tf_dyn.model.model_var))
    print('OBS: {}'.format(alg['bsqkf'].tf_meas.model.model_var.diagonal()))
    print('LBSQKF Expected Model Variance')
    print('OBS: {}'.format(alg['lbsqkf'].tf_meas.model.model_var.diagonal()))
    print()

    # compute performance scores
    num_alg = len(alg)
    d, steps, mc_sims = x.shape
    mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
    for ia, a in enumerate(alg):
        print('Running {:<6} ... '.format(a.upper()), end='', flush=True)
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
    for a in range(num_alg):
        for k in range(steps):
            mse = mse_matrix(x[:, k, :], mean[:, k, :, a])
            for imc in range(mc_sims):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                cov_sel = cov[..., k, imc, a]
                lcr_pos[k, imc, a] = log_cred_ratio(x[(0, 2), k, imc], mean[(0, 2), k, imc, a],
                                                    cov_sel[np.ix_((0, 2), (0, 2))], mse[np.ix_((0, 2), (0, 2))])
                lcr_vel[k, imc, a] = log_cred_ratio(x[(1, 3), k, imc], mean[(1, 3), k, imc, a],
                                                    cov_sel[np.ix_((1, 3), (1, 3))], mse[np.ix_((1, 3), (1, 3))])

    # Averaged RMSE and Inclination Indicator in time
    pos_rmse_vs_time = np.sqrt((error2[(0, 2), ...]).sum(axis=0)).mean(axis=1)
    vel_rmse_vs_time = np.sqrt((error2[(1, 3), ...]).sum(axis=0)).mean(axis=1)
    pos_inc_vs_time = lcr_pos.mean(axis=1)
    vel_inc_vs_time = lcr_vel.mean(axis=1)

    # print performance scores
    row_labels = [k.upper() for k in alg.keys()]
    col_labels = ['RMSE', 'Inc']

    pos_data = np.vstack((pos_rmse_vs_time.mean(axis=0), pos_inc_vs_time.mean(axis=0))).T
    pos_table = pd.DataFrame(pos_data, index=row_labels, columns=col_labels)
    print(pos_table)

    vel_data = np.vstack((vel_rmse_vs_time.mean(axis=0), vel_inc_vs_time.mean(axis=0))).T
    vel_table = pd.DataFrame(vel_data, index=row_labels, columns=col_labels)
    print(vel_table)

    # PLOTS # TODO: figures for states rather than scores
    # root mean squared error
    plt.figure().suptitle('RMSE')
    g = GridSpec(2, 1)
    ax = plt.subplot(g[0, 0])
    ax.set_title('Position')
    for i, f_str in enumerate(alg):
        ax.plot(pos_rmse_vs_time[:, i], label=f_str.upper())
    ax.legend()
    ax = plt.subplot(g[1, 0])
    ax.set_title('Velocity')
    for i, f_str in enumerate(alg):
        ax.plot(vel_rmse_vs_time[:, i])
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
    plt.show()


if __name__ == '__main__':
    # reentry_simple_demo()
    # reentry_demo()
    # coordinated_turn_radar_demo()
    constant_velocity_radar_tracking_demo(steps=200, mc_sims=50)
