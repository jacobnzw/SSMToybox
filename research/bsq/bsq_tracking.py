import numpy as np
import pandas as pd
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve
from journal_figure import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from collections import OrderedDict
import joblib
import time
import os

from ssmtoybox.ssinf import GaussianProcessKalman, BayesSardKalman, UnscentedKalman, ExtendedKalman, GaussianInference
from ssmtoybox.ssmod import ReentryVehicle1DTransition, ReentryVehicle2DTransition, CoordinatedTurnTransition, \
    ConstantTurnRateSpeed, RangeMeasurement, BearingMeasurement, Radar2DMeasurement
from ssmtoybox.utils import mse_matrix, squared_error, log_cred_ratio, GaussRV
from ssmtoybox.ssmod import TransitionModel, MeasurementModel
from ssmtoybox.bq.bqmtran import BayesSardTransform
from ssmtoybox.mtran import LinearizationTransform


np.set_printoptions(precision=4)


# def reentry_simple_demo(dur=30, tau=0.1, mc=100):
#     """
#     A spherical object falls down from high altitude entering the Earth’s atmosphere with a high velocity.
#
#     Parameters
#     ----------
#     tau: float
#         discretization period for the dynamics ODE integration method
#     dur: int
#         Duration of the dynamics simulation
#     mc: int
#         Number of Monte Carlo simulations.
#
#     Notes
#     -----
#     The parameter mc determines the number of trajectories simulated.
#
#     Returns
#     -------
#     """
#
#     # Generate reference trajectory by ODE integration
#     sys = ReentryVehicleRadarTrackingSimpleGaussSystem()
#     x = sys.simulate_trajectory(method='rk4', dt=tau, duration=dur, mc_sims=mc)
#
#     # pick only non-divergent trajectories
#     x = x[..., np.all(x >= 0, axis=(0, 1))]
#     mc = x.shape[2]
#
#     y = np.zeros((sys.zD,) + x.shape[1:])
#     for i in range(mc):
#         y[..., i] = sys.simulate_measurements(x[..., i], mc_per_step=1).squeeze()
#
#     # Initialize model
#     ssm = ReentryVehicleRadarTrackingSimpleGaussSSM(dt=tau)
#
#     # GPQKF kernel parameters
#     kpar_dyn_ut = np.array([[0.5, 10, 10, 10]])
#     kpar_obs_ut = np.array([[0.5, 15, 20, 20]])
#
#     # UT multi-index
#     mul_ut = np.hstack((np.zeros((ssm.xD, 1)), np.eye(ssm.xD), 2*np.eye(ssm.xD))).astype(np.int)
#
#     # Initialize filters
#     alg = (
#         GaussianProcessKalman(ssm, kpar_dyn_ut, kpar_obs_ut, kernel='rbf', points='ut'),
#         BayesSardKalman(ssm, kpar_dyn_ut, kpar_obs_ut, mul_ut, mul_ut, points='ut'),
#         UnscentedKalman(ssm, beta=0),
#     )
#
#     num_alg = len(alg)
#     d, steps, mc = x.shape
#     mean, cov = np.zeros((d, steps, mc, num_alg)), np.zeros((d, d, steps, mc, num_alg))
#     for imc in range(mc):
#         for ia, a in enumerate(alg):
#             # Do filtering and reset the filters for each new track
#             mean[..., imc, ia], cov[..., imc, ia] = a.forward_pass(y[..., imc])
#             a.reset()
#
#     # time index for plotting
#     time_ind = np.linspace(1, dur, x.shape[1])
#
#     # PLOTS: Trajectories
#     # plt.figure()
#     # g = GridSpec(4, 2)
#     # plt.subplot(g[:2, :])
#     #
#     # # Earth surface w/ radar position
#     # t = np.arange(0.48 * np.pi, 0.52 * np.pi, 0.01)
#     # plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t) - sys.R0, 'darkblue', lw=2)
#     # plt.plot(sys.sx, sys.sy, 'ko')
#     #
#     # xzer = np.zeros(x.shape[1])
#     # for i in range(mc):
#     #     # Vehicle trajectory
#     #     plt.plot(xzer, x[0, :, i], alpha=0.35, color='r', ls='--', lw=2)
#     #
#     #     # Filtered position estimate
#     #     plt.plot(xzer, mean[0, :, i, 0], color='g', alpha=0.3)
#     #     plt.plot(xzer, mean[0, :, i, 1], color='orange', alpha=0.3)
#
#     # Altitude
#     # x0 = sys.pars['x0_mean']
#     # plt.subplot(g[2, :])
#     # plt.ylim([0, x0[0]])
#     # for i in range(mc):
#     #     plt.plot(time_ind, x[0, :, i], alpha=0.35, color='b')
#     # plt.ylabel('altitude [ft]')
#     # plt.xlabel('time [s]')
#     #
#     # # Velocity
#     # plt.subplot(g[3, :])
#     # plt.ylim([0, x0[1]])
#     # for i in range(mc):
#     #     plt.plot(time_ind, x[1, :, i], alpha=0.35, color='b')
#     # plt.ylabel('velocity [ft/s]')
#     # plt.xlabel('time [s]')
#
#     # Compute Performance Scores
#     error2 = mean.copy()
#     pos_lcr = np.zeros((steps, mc, num_alg))
#     vel_lcr = pos_lcr.copy()
#     theta_lcr = pos_lcr.copy()
#     for a in range(num_alg):
#         for k in range(steps):
#             pos_mse = mse_matrix(x[0, na, k, :], mean[0, na, k, :, a])
#             vel_mse = mse_matrix(x[1, na, k, :], mean[1, na, k, :, a])
#             theta_mse = mse_matrix(x[2, na, k, :], mean[2, na, k, :, a])
#             for imc in range(mc):
#                 error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
#                 pos_lcr[k, imc, a] = log_cred_ratio(x[0, k, imc], mean[0, k, imc, a],
#                                                     cov[0, 0, k, imc, a], pos_mse)
#                 vel_lcr[k, imc, a] = log_cred_ratio(x[1, k, imc], mean[1, k, imc, a],
#                                                     cov[1, 1, k, imc, a], vel_mse)
#                 theta_lcr[k, imc, a] = log_cred_ratio(x[2, k, imc], mean[2, k, imc, a],
#                                                       cov[2, 2, k, imc, a], theta_mse)
#
#     # Averaged position/velocity RMSE and inclination in time
#     pos_rmse = np.sqrt(error2[0, na, ...].sum(axis=0))
#     pos_rmse_vs_time = pos_rmse.mean(axis=1)
#     pos_inc_vs_time = pos_lcr.mean(axis=1)
#     vel_rmse = np.sqrt(error2[1, na, ...].sum(axis=0))
#     vel_rmse_vs_time = vel_rmse.mean(axis=1)
#     vel_inc_vs_time = vel_lcr.mean(axis=1)
#     theta_rmse = np.sqrt(error2[2, na, ...].sum(axis=0))
#     theta_rmse_vs_time = theta_rmse.mean(axis=1)
#     theta_inc_vs_time = theta_lcr.mean(axis=1)
#
#     # PLOTS: RMSE in time for position, velocity and ballistic parameter
#     plt.figure()
#     g = GridSpec(6, 3)
#     # filt_labels = ['UT', 'CUT', 'GPQ-UT', 'GPQ-CUT']
#     filt_labels = ['GPQKF', 'BSQKF', 'UKF']
#     plt.subplot(g[:2, :2])
#     plt.ylabel('RMSE')
#     for i in range(num_alg):
#         plt.plot(time_ind[1:], pos_rmse_vs_time[1:, i], label=filt_labels[i])
#     plt.legend()
#
#     plt.subplot(g[2:4, :2])
#     plt.ylabel('RMSE')
#     for i in range(num_alg):
#         plt.plot(time_ind[1:], vel_rmse_vs_time[1:, i], label=filt_labels[i])
#
#     plt.subplot(g[4:, :2])
#     plt.ylabel('RMSE')
#     plt.xlabel('time step [k]')
#     for i in range(num_alg):
#         plt.plot(time_ind[1:], theta_rmse_vs_time[1:, i], label=filt_labels[i])
#
#     # BOX PLOTS: time-averaged RMSE
#     plt.subplot(g[:2, 2:])
#     plt.boxplot(pos_rmse.mean(axis=0), labels=filt_labels)
#
#     plt.subplot(g[2:4, 2:])
#     plt.boxplot(vel_rmse.mean(axis=0), labels=filt_labels)
#
#     plt.subplot(g[4:, 2:])
#     plt.boxplot(theta_rmse.mean(axis=0), labels=filt_labels)
#     plt.show()
#
#     # PLOTS: Inclination indicator in time for position, velocity and ballistic parameter
#     plt.figure()
#     g = GridSpec(6, 3)
#
#     plt.subplot(g[:2, :2])
#     plt.ylabel(r'$\nu$')
#     for i in range(num_alg):
#         plt.plot(time_ind, pos_inc_vs_time[:, i], label=filt_labels[i])
#
#     plt.subplot(g[2:4, :2])
#     plt.ylabel(r'$\nu$')
#     for i in range(num_alg):
#         plt.plot(time_ind, vel_inc_vs_time[:, i], label=filt_labels[i])
#
#     plt.subplot(g[4:, :2])
#     plt.ylabel(r'$\nu$')
#     plt.xlabel('time step [k]')
#     for i in range(num_alg):
#         plt.plot(time_ind, theta_inc_vs_time[:, i], label=filt_labels[i])
#
#     # BOX PLOTS: time-averaged inclination indicator
#     plt.subplot(g[:2, 2:])
#     plt.boxplot(pos_lcr.mean(axis=0), labels=filt_labels)
#
#     plt.subplot(g[2:4, 2:])
#     plt.boxplot(vel_lcr.mean(axis=0), labels=filt_labels)
#
#     plt.subplot(g[4:, 2:])
#     plt.boxplot(theta_lcr.mean(axis=0), labels=filt_labels)
#     plt.show()
#
#     # TODO: pandas tables for printing into latex
#     np.set_printoptions(precision=4)
#     print('Average RMSE: {}'.format(np.sqrt(error2.sum(axis=0)).mean(axis=(0, 1))))


def reentry_demo(dur=200, mc_sims=100, outfile=None, regenerate=False):
    # use default filename if unspecified
    if outfile is None:
        outfile = 'reentry_demo_results.dat'
    outfile = os.path.join(outfile)

    if not os.path.exists(outfile) or regenerate:
        tau = 0.05
        disc_tau = 0.1

        # initial condition statistics
        m0 = np.array([6500, 350, -1.8, -6.8, 0.7])
        P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 0])
        init_rv = GaussRV(5, m0, P0)

        # process noise statistics
        noise_rv = GaussRV(3, cov=np.diag([2.4e-5, 2.4e-5, 0]))
        sys = ReentryVehicle2DTransition(init_rv, noise_rv)

        # measurement noise statistics
        meas_noise_rv = GaussRV(2, cov=np.diag([1e-6, 0.17e-6]))
        obs = Radar2DMeasurement(meas_noise_rv, 5, radar_loc=np.array([6374, 0.0]))

        np.random.seed(0)
        # Generate reference trajectory by SDE simulation
        x = sys.simulate_continuous(duration=dur, dt=tau, mc_sims=mc_sims)
        y = obs.simulate_measurements(x)

        # subsample data for the filter and performance score calculation
        x = x[:, ::2, ...]  # take every second point on the trajectory
        y = y[:, ::2, ...]  # and corresponding measurement

        # Initialize state-space model with mis-specified initial mean
        # m0 = np.array([6500.4, 349.14, -1.1093, -6.1967, 0.6932])
        m0 = np.array([6500, 350, -1.1, -6.1, 0.7])
        P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
        init_rv = GaussRV(5, m0, P0)
        noise_rv = GaussRV(3, cov=np.diag([2.4e-5, 2.4e-5, 1e-6]))
        dyn = ReentryVehicle2DTransition(init_rv, noise_rv, dt=disc_tau)
        # NOTE: higher tau causes higher spread of trajectories at the ends

        # Initialize filters
        par_dyn = np.array([[1.0, 1, 1, 1, 1, 1]])
        par_obs = np.array([[1.0, 0.9, 0.9, 1e4, 1e4, 1e4]])  # np.atleast_2d(np.ones(6))
        mul_ut = np.hstack((np.zeros((dyn.dim_in, 1)), np.eye(dyn.dim_in), 2 * np.eye(dyn.dim_in))).astype(np.int)
        alg = OrderedDict({
            # GaussianProcessKalman(ssm, par_dyn, par_obs, kernel='rbf', points='ut'),
            'bsqkf': BayesSardKalman(dyn, obs, par_dyn, par_obs, mul_ut, mul_ut, points='ut'),
            'bsqkf_2e-6': BayesSardKalman(dyn, obs, par_dyn, par_obs, mul_ut, mul_ut, points='ut'),
            'bsqkf_2e-7': BayesSardKalman(dyn, obs, par_dyn, par_obs, mul_ut, mul_ut, points='ut'),
            'ukf': UnscentedKalman(dyn, obs, beta=0.0),
        })

        alg['bsqkf'].tf_dyn.model.model_var = np.diag([0.0002, 0.0002, 0.0002, 0.0002, 0.0002])
        alg['bsqkf'].tf_obs.model.model_var = 0*np.eye(2)
        alg['bsqkf_2e-6'].tf_dyn.model.model_var = 2e-6*np.eye(5)
        alg['bsqkf_2e-6'].tf_obs.model.model_var = 0 * np.eye(2)
        alg['bsqkf_2e-7'].tf_dyn.model.model_var = 2e-7*np.eye(5)
        alg['bsqkf_2e-7'].tf_obs.model.model_var = 0 * np.eye(2)

        # print experiment setup
        print('System (reality)')
        print('{:10s}: {}'.format('x0_mean', sys.init_rv.mean))
        print('{:10s}: {}'.format('x0_cov', sys.init_rv.cov.diagonal()))
        print()
        print('State-Space Model')
        print('{:10s}: {}'.format('x0_mean', dyn.init_rv.mean))
        print('{:10s}: {}'.format('x0_cov', dyn.init_rv.cov.diagonal()))
        print()
        print('BSQ Expected Model Variance')
        print('{:10s}: {}'.format('dyn_func', alg['bsqkf'].tf_dyn.model.model_var.diagonal()))
        print('{:10s}: {}'.format('obs_func', alg['bsqkf'].tf_obs.model.model_var.diagonal()))
        print()

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
        lcr_x = np.zeros((steps, mc_sims, num_alg))
        lcr_pos = lcr_x.copy()
        lcr_vel = lcr_x.copy()
        lcr_theta = lcr_x.copy()
        for a in range(num_alg):
            for k in range(steps):
                mse = mse_matrix(x[:, k, :], mean[:, k, :, a])
                for imc in range(mc_sims):
                    error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                    lcr_x[k, imc, a] = log_cred_ratio(x[:, k, imc], mean[:, k, imc, a], cov[:, :, k, imc, a], mse)
                    lcr_pos[k, imc, a] = log_cred_ratio(x[:2, k, imc], mean[:2, k, imc, a],
                                                        cov[:2, :2, k, imc, a], mse[:2, :2])
                    lcr_vel[k, imc, a] = log_cred_ratio(x[2:4, k, imc], mean[2:4, k, imc, a],
                                                        cov[2:4, 2:4, k, imc, a], mse[2:4, 2:4])
                    lcr_theta[k, imc, a] = log_cred_ratio(x[4, k, imc], mean[4, k, imc, a],
                                                          cov[4, 4, k, imc, a], mse[4, 4])

        # Averaged RMSE and Inclination Indicator in time
        x_rmse_vs_time = np.sqrt(error2.sum(axis=0)).mean(axis=1)
        pos_rmse_vs_time = np.sqrt((error2[:2, ...]).sum(axis=0)).mean(axis=1)
        vel_rmse_vs_time = np.sqrt((error2[2:4, ...]).sum(axis=0)).mean(axis=1)
        theta_rmse_vs_time = np.sqrt((error2[4, ...])).mean(axis=1)
        x_inc_vs_time = lcr_x.mean(axis=1)
        pos_inc_vs_time = lcr_pos.mean(axis=1)
        vel_inc_vs_time = lcr_vel.mean(axis=1)
        theta_inc_vs_time = lcr_theta.mean(axis=1)

        print('{:10s}: {}'.format('RMSE', x_rmse_vs_time.mean(axis=0)))
        print('{:10s}: {}'.format('INC', x_inc_vs_time.mean(axis=0)))

        # Save the simulation results into outfile
        data_dict = {
            'duration': dur,
            'disc_tau': disc_tau,
            'alg_str': list(alg.keys()),
            'x': x,
            'mean': mean,
            'cov': cov,
            'state': {'rmse': x_rmse_vs_time, 'inc': x_inc_vs_time},
            'position': {'rmse': pos_rmse_vs_time, 'inc': pos_inc_vs_time},
            'velocity': {'rmse': vel_rmse_vs_time, 'inc': vel_inc_vs_time},
            'parameter': {'rmse': theta_rmse_vs_time, 'inc': theta_inc_vs_time},
        }
        joblib.dump(data_dict, outfile)

        # Visualize simulation results, plots, tables
        reentry_demo_results(data_dict)
    else:
        reentry_demo_results(joblib.load(outfile))


def reentry_demo_results(data_dict):
    alg_str = data_dict['alg_str']
    # PLOTS
    steps = data_dict['position']['rmse'].shape[0]
    state_labels = ['Position', 'Velocity', 'Parameter']
    time_sec = np.linspace(0, data_dict['duration']-data_dict['disc_tau'], steps)

    plt.style.use('seaborn-deep')
    printfig = FigurePrint()
    for state_label in state_labels:
        fig, ax = plt.subplots(2, 1, sharex=True)
        # fig.suptitle(state_label)
        for i, f_str in enumerate(alg_str):
            ax[0].plot(time_sec, data_dict[state_label.lower()]['rmse'][:, i], label=f_str.upper(), lw=2)

        # decide which units based on state components
        # TODO: Can be avoided by using OrderedDict for state_labels
        if state_label == 'Position':
            units = r' [$\si{km}$]'
        elif state_label == 'Velocity':
            units = r' [$\si{km.s^{-1}}$]'
        else:
            units = r' [-]'

        ax[0].set_ylabel('RMSE' + units)
        ax[0].legend()
        for i, f_str in enumerate(alg_str):
            ax[1].plot(time_sec, data_dict[state_label.lower()]['inc'][:, i], label=f_str.upper(), lw=2)
        ax[1].add_line(Line2D([0, time_sec[-1]], [0, 0], linewidth=2, color='k', ls='--'))
        ax[1].set_ylabel('INC [-]')
        ax[1].set_xlabel(r'time [$\si{s}$]')
        plt.tight_layout(pad=0)
    plt.show()

    for i, sl in zip(plt.get_fignums(), state_labels):
        plt.figure(i)
        printfig.savefig('radar_tracking_{:s}'.format(sl.lower()))


def reentry_demo_table(data_dict):
    state_labels = ['position', 'velocity', 'parameter']
    score_labels = ['rmse', 'inc']
    col_labels = ['rmse (bsqkf)', 'rmse (ukf)', 'inc (bsqkf)', 'inc (ukf)']
    table_data = np.zeros((len(state_labels), len(score_labels), 2))
    for st_idx, st_label in enumerate(state_labels):
        for sc_idx, sc_label in enumerate(score_labels):
            table_data[st_idx, sc_idx] = data_dict[st_label][sc_label].mean(axis=0)

    table_data = table_data.reshape(len(state_labels), 2*len(score_labels))
    tab = pd.DataFrame(table_data, index=state_labels, columns=col_labels)
    pd.set_option('precision', 2)
    print(tab.to_latex())


# def coordinated_turn_radar_demo():
#     num_steps = 100
#     mc_sims = 20
#
#     # sensor positions
#     x_min, x_max = -10000, 10000
#     y_min, y_max = -10000, 10000
#     S = np.array([[x_min, y_min],
#                   [x_min, y_max],
#                   [x_max, y_min],
#                   [x_max, y_max]])
#     tau = 1.0
#     # generate data
#     sys = CoordinatedTurnBearingsOnlyTrackingGaussSSM(dt=tau, sensor_pos=S)
#     x, y = sys.simulate(num_steps, mc_sims)
#
#     # weed out trajectories venturing outside of the sensor rectangle
#     ix = np.all(np.abs(x[(0, 2), ...]) <= x_max, axis=(0, 1))
#     x, y = x[..., ix], y[..., ix]
#     print('{:.2f}% of trajectories weeded out.'.format(100 * np.count_nonzero(ix == False) / len(ix)))
#
#     # state-space model for filters
#     ssm = CoordinatedTurnBearingsOnlyTrackingGaussSSM(dt=tau, sensor_pos=S)
#     # Initialize filters
#     par_dyn = np.array([[1.0, 1, 1, 1, 1, 1]])
#     par_obs = np.array([[1.0, 0.9, 0.9, 1e4, 1e4, 1e4]])
#     mul_ut = np.hstack((np.zeros((ssm.xD, 1)), np.eye(ssm.xD), 2 * np.eye(ssm.xD))).astype(np.int)
#     alg = OrderedDict({
#         # GaussianProcessKalman(ssm, par_dyn, par_obs, kernel='rbf', points='ut'),
#         'bsqkf': BayesSardKalman(ssm, par_dyn, par_obs, mul_ut, mul_ut, points='ut'),
#         'ukf': UnscentedKalman(ssm, beta=0),
#     })
#
#     kpdyn = np.array([[1.0, 1, 1e3, 1, 1e3, 1e3],
#                       [1.0, 1e3, 1, 1e3, 1, 1e3],
#                       [1.0, 3, 3, 3, 3, 1],
#                       [1.0, 3, 3, 3, 3, 1],
#                       [1.0, 1e3, 1e3, 1e3, 1e3, 1]])
#     alg['bsqkf'].tf_dyn.model.model_var = 1e-3*np.eye(5)  # multivariate_emv(alg['bsqkf'].tf_dyn, kpdyn, mul_ut)
#     kpobs = np.array([[1.0, 1, 1, 1e2, 1e2, 1e2],
#                       [1.0, 1.4, 1.4, 1e2, 1e2, 1e2]])
#     # multivariate_emv(alg[0].tf_meas, kpobs, mul_ut)  # 1e-8*np.eye(2)
#     alg['bsqkf'].tf_meas.model.model_var = 1e-3*np.eye(4)
#     print('BSQ EMV\ndyn: {} \nobs: {}'.format(alg['bsqkf'].tf_dyn.model.model_var.diagonal(),
#                                               alg['bsqkf'].tf_meas.model.model_var.diagonal()))
#
#     num_alg = len(alg)
#     d, steps, mc_sims = x.shape
#     mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
#     for ia, a in enumerate(alg):
#         print('Running {:<5} ... '.format(a.upper()), end='', flush=True)
#         t0 = time.time()
#         for imc in range(mc_sims):
#             mean[..., imc, ia], cov[..., imc, ia] = alg[a].forward_pass(y[..., imc])
#             alg[a].reset()
#         print('{:>30}'.format('Done in {:.2f} [sec]'.format(time.time() - t0)))
#
#     # Performance score plots
#     print()
#     print('Computing performance scores ...')
#     error2 = mean.copy()
#     lcr_pos = np.zeros((steps, mc_sims, num_alg))
#     lcr_vel = lcr_pos.copy()
#     lcr_theta = lcr_pos.copy()
#     for a in range(num_alg):
#         for k in range(steps):
#             mse = mse_matrix(x[:, k, :], mean[:, k, :, a])
#             for imc in range(mc_sims):
#                 error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
#                 cov_sel = cov[..., k, imc, a]
#                 lcr_pos[k, imc, a] = log_cred_ratio(x[(0, 2), k, imc], mean[(0, 2), k, imc, a],
#                                                     cov_sel[np.ix_((0, 2), (0, 2))], mse[np.ix_((0, 2), (0, 2))])
#                 lcr_vel[k, imc, a] = log_cred_ratio(x[(1, 3), k, imc], mean[(1, 3), k, imc, a],
#                                                     cov_sel[np.ix_((1, 3), (1, 3))], mse[np.ix_((1, 3), (1, 3))])
#                 lcr_theta[k, imc, a] = log_cred_ratio(x[4, k, imc], mean[4, k, imc, a],
#                                                       cov[4, 4, k, imc, a], mse[4, 4])
#
#     # Averaged RMSE and Inclination Indicator in time
#     pos_rmse_vs_time = np.sqrt((error2[(0, 2), ...]).sum(axis=0)).mean(axis=1)
#     vel_rmse_vs_time = np.sqrt((error2[(1, 3), ...]).sum(axis=0)).mean(axis=1)
#     theta_rmse_vs_time = np.sqrt((error2[4, ...])).mean(axis=1)
#     pos_inc_vs_time = lcr_pos.mean(axis=1)
#     vel_inc_vs_time = lcr_vel.mean(axis=1)
#     theta_inc_vs_time = lcr_theta.mean(axis=1)
#
#     # print performance scores
#     print('Average position RMSE: {}'.format(pos_rmse_vs_time.mean(axis=0)))
#     print('Average position Inc.: {}'.format(pos_inc_vs_time.mean(axis=0)))
#     print('Average velocity RMSE: {}'.format(vel_rmse_vs_time.mean(axis=0)))
#     print('Average velocity Inc.: {}'.format(vel_inc_vs_time.mean(axis=0)))
#     print('Average parameter RMSE: {}'.format(theta_rmse_vs_time.mean(axis=0)))
#     print('Average parameter Inc.: {}'.format(theta_inc_vs_time.mean(axis=0)))
#
#     # PLOTS # TODO: figures for states rather than scores
#     # root mean squared error
#     plt.figure().suptitle('RMSE')
#     g = GridSpec(3, 1)
#     ax = plt.subplot(g[0, 0])
#     ax.set_title('Position')
#     for i, f_str in enumerate(alg):
#         ax.plot(pos_rmse_vs_time[:, i], label=f_str.upper())
#     ax.legend()
#     ax = plt.subplot(g[1, 0])
#     ax.set_title('Velocity')
#     for i, f_str in enumerate(alg):
#         ax.plot(vel_rmse_vs_time[:, i])
#     ax = plt.subplot(g[2, 0])
#     ax.set_title('Parameter')
#     for i, f_str in enumerate(alg):
#         ax.plot(theta_rmse_vs_time[:, i])
#     plt.show(block=False)
#
#     # inclination
#     plt.figure().suptitle('Inclination')
#     ax = plt.subplot(g[0, 0])
#     ax.set_title('Position')
#     for i, f_str in enumerate(alg):
#         ax.plot(pos_inc_vs_time[:, i], label=f_str.upper())
#     ax.legend()
#     ax = plt.subplot(g[1, 0])
#     ax.set_title('Velocity')
#     for i, f_str in enumerate(alg):
#         ax.plot(vel_inc_vs_time[:, i])
#     ax = plt.subplot(g[2, 0])
#     ax.set_title('Parameter')
#     for i, f_str in enumerate(alg):
#         ax.plot(theta_inc_vs_time[:, i])
#     plt.show()


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


# class ConstantVelocityRadarTrackingGaussSSM(GaussianStateSpaceModel):
#     xD = 4
#     zD = 2
#     qD = 2
#     rD = 2
#
#     q_additive = True
#     r_additive = True
#
#     def __init__(self, dt=0.5):
#         self.dt = dt
#         self.q_gain = np.array([[dt ** 2 / 2, 0],
#                                 [dt, 0],
#                                 [0, dt ** 2 / 2],
#                                 [0, dt]])
#         pars = {
#             'x0_mean': np.array([10175, 295, 980, -35]),  # m, m/s, m, m/s
#             'x0_cov': np.diag([100**2, 10**2, 100**2, 10**2]),
#             'q_mean': np.zeros((self.qD, )),
#             'q_cov': np.diag([50, 5]),  # m^2/s^4, m^2/s^4
#             'q_gain': self.q_gain,
#             'r_mean': np.zeros((self.rD, )),
#             # 'r_cov': np.diag([50, 0.4]),  # m^2, mrad^2
#             'r_cov': np.diag([50, 0.4e-6]),  # m^2, rad^2
#         }
#         super(ConstantVelocityRadarTrackingGaussSSM, self).__init__(**pars)
#
#     def dyn_fcn(self, x, q, pars):
#         A = np.array([[1, self.dt, 0, 0],
#                       [0, 1, 0, 0],
#                       [0, 0, 1, self.dt],
#                       [0, 0, 0, 1]])
#         return A.dot(x) + self.q_gain.dot(q)
#
#     def meas_fcn(self, x, r, pars):
#         rang = np.sqrt(x[0]**2 + x[2]**2)
#         theta = np.arctan2(x[2], x[0])
#         return np.array([rang, theta]) + r
#
#     def par_fcn(self, time):
#         pass
#
#     def dyn_fcn_dx(self, x, q, pars):
#         return np.array([[1, self.dt, 0, 0],
#                          [0, 1, 0, 0],
#                          [0, 0, 1, self.dt],
#                          [0, 0, 0, 1]]).T
#
#     def meas_fcn_dx(self, x, r, pars):
#         rang = np.sqrt(x[0] ** 2 + x[2] ** 2)
#         return np.array([[x[0]/rang, 0, x[2]/rang, 0],
#                          [-x[2]/(rang**2), 0, x[0]/(rang**2), 0]])


class LinearBayesSardKalman(GaussianInference):
    """
    Gaussian filter with linear MT on the dynamics model and BSQ-MT on observation model.
    """
    def __init__(self, dyn, obs, kern_par_obs, mulind_obs=2, points='ut', point_hyp=None):
        t_dyn = LinearizationTransform(dyn.dim_in)

        rbf_kernel = {'name': 'rbf', 'params': kern_par_obs}
        point_spec = {'name': points, 'params': point_hyp}
        t_obs = BayesSardTransform(obs.dim_in, obs.dim_out, rbf_kernel, point_spec, mulind_obs)

        super(LinearBayesSardKalman, self).__init__(dyn, obs, t_dyn, t_obs)


# def constant_velocity_radar_tracking_demo(steps=100, mc_sims=100):
#
#     sys = ConstantVelocityRadarTrackingGaussSSM()
#     sys.set_pars('x0_mean', np.array([10000, 300, 1000, -40]))
#     x, y = sys.simulate(steps, mc_sims)
#
#     sys.check_jacobians(h=np.sqrt(np.finfo(float).eps))
#
#     # state-space model for filters
#     ssm = ConstantVelocityRadarTrackingGaussSSM()
#     # ssm.set_pars('x0_mean', np.array([10000, 300, 1000, -40]))
#     # kernel parameters for dynamics and observation functions
#     kpar_dyn = np.array([[1.0] + ssm.xD * [10]])
#     kpar_obs = np.array([[1.0] + ssm.xD * [1]])
#     alpha_ut = np.hstack((np.zeros((ssm.xD, 1)), np.eye(ssm.xD), 2*np.eye(ssm.xD))).astype(np.int)
#     alg = OrderedDict({
#         'bsqkf': BayesSardKalman(ssm, kpar_dyn, kpar_obs, alpha_ut, alpha_ut, points='ut'),
#         'lbsqkf': LinearBayesSardKalman(ssm, kpar_obs, alpha_ut, points='ut'),
#         'ukf': UnscentedKalman(ssm),
#         # 'ekf': ExtendedKalman(ssm),
#     })
#
#     # set custom model variance
#     kpobs = np.array([[0.00001, 1, 2, 1e2, 1e2],
#                       [0.00001, 1, 2, 1e2, 1e2]])
#     # multivariate_emv(alg['bsqkf'].tf_meas, kpobs, alpha_ut)  # 1e-3 * np.eye(4)
#     # multivariate_emv(alg['lbsqkf'].tf_meas, kpobs, alpha_ut)
#     alg['bsqkf'].tf_meas.model.model_var = 1e-8 * np.eye(2)
#     alg['lbsqkf'].tf_meas.model.model_var = 1e-8 * np.eye(2)
#
#     print('BSQKF Expected Model Variance')
#     print('DYN: {}'.format(alg['bsqkf'].tf_dyn.model.model_var))
#     print('OBS: {}'.format(alg['bsqkf'].tf_meas.model.model_var.diagonal()))
#     print('LBSQKF Expected Model Variance')
#     print('OBS: {}'.format(alg['lbsqkf'].tf_meas.model.model_var.diagonal()))
#     print()
#
#     # compute performance scores
#     num_alg = len(alg)
#     d, steps, mc_sims = x.shape
#     mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
#     for ia, a in enumerate(alg):
#         print('Running {:<6} ... '.format(a.upper()), end='', flush=True)
#         t0 = time.time()
#         for imc in range(mc_sims):
#             mean[..., imc, ia], cov[..., imc, ia] = alg[a].forward_pass(y[..., imc])
#             alg[a].reset()
#         print('{:>30}'.format('Done in {:.2f} [sec]'.format(time.time() - t0)))
#
#     # Performance score plots
#     print()
#     print('Computing performance scores ...')
#     error2 = mean.copy()
#     lcr_pos = np.zeros((steps, mc_sims, num_alg))
#     lcr_vel = lcr_pos.copy()
#     for a in range(num_alg):
#         for k in range(steps):
#             mse = mse_matrix(x[:, k, :], mean[:, k, :, a])
#             for imc in range(mc_sims):
#                 error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
#                 cov_sel = cov[..., k, imc, a]
#                 lcr_pos[k, imc, a] = log_cred_ratio(x[(0, 2), k, imc], mean[(0, 2), k, imc, a],
#                                                     cov_sel[np.ix_((0, 2), (0, 2))], mse[np.ix_((0, 2), (0, 2))])
#                 lcr_vel[k, imc, a] = log_cred_ratio(x[(1, 3), k, imc], mean[(1, 3), k, imc, a],
#                                                     cov_sel[np.ix_((1, 3), (1, 3))], mse[np.ix_((1, 3), (1, 3))])
#
#     # Averaged RMSE and Inclination Indicator in time
#     pos_rmse_vs_time = np.sqrt((error2[(0, 2), ...]).sum(axis=0)).mean(axis=1)
#     vel_rmse_vs_time = np.sqrt((error2[(1, 3), ...]).sum(axis=0)).mean(axis=1)
#     pos_inc_vs_time = lcr_pos.mean(axis=1)
#     vel_inc_vs_time = lcr_vel.mean(axis=1)
#
#     # print performance scores
#     row_labels = [k.upper() for k in alg.keys()]
#     col_labels = ['RMSE', 'Inc']
#
#     pos_data = np.vstack((pos_rmse_vs_time.mean(axis=0), pos_inc_vs_time.mean(axis=0))).T
#     pos_table = pd.DataFrame(pos_data, index=row_labels, columns=col_labels)
#     print(pos_table)
#
#     vel_data = np.vstack((vel_rmse_vs_time.mean(axis=0), vel_inc_vs_time.mean(axis=0))).T
#     vel_table = pd.DataFrame(vel_data, index=row_labels, columns=col_labels)
#     print(vel_table)
#
#     # PLOTS # TODO: figures for states rather than scores
#     # root mean squared error
#     plt.figure().suptitle('RMSE')
#     g = GridSpec(2, 1)
#     ax = plt.subplot(g[0, 0])
#     ax.set_title('Position')
#     for i, f_str in enumerate(alg):
#         ax.plot(pos_rmse_vs_time[:, i], label=f_str.upper())
#     ax.legend()
#     ax = plt.subplot(g[1, 0])
#     ax.set_title('Velocity')
#     for i, f_str in enumerate(alg):
#         ax.plot(vel_rmse_vs_time[:, i])
#     plt.show(block=False)
#
#     # inclination
#     plt.figure().suptitle('Inclination')
#     ax = plt.subplot(g[0, 0])
#     ax.set_title('Position')
#     for i, f_str in enumerate(alg):
#         ax.plot(pos_inc_vs_time[:, i], label=f_str.upper())
#     ax.legend()
#     ax = plt.subplot(g[1, 0])
#     ax.set_title('Velocity')
#     for i, f_str in enumerate(alg):
#         ax.plot(vel_inc_vs_time[:, i])
#     plt.show()


# class ConstantTurnRateAndVelocityGaussSSM(GaussianStateSpaceModel):
#
#     xD = 5
#     zD = 2
#     qD = 2
#     rD = 2
#
#     q_additive = False
#     r_additive = True
#
#     def __init__(self, dt=0.5):
#         stats = {
#             'x0_mean': np.zeros((self.xD, )),
#             'x0_cov': 0.1*np.eye(self.xD),
#             'q_mean': np.zeros((self.qD, )),
#             'q_cov': np.diag([0.1, 0.1*np.pi]),
#             'r_mean': np.zeros((self.rD, )),
#             'r_cov': np.diag([0.3, 0.03]),
#         }
#         super(ConstantTurnRateAndVelocityGaussSSM, self).__init__(**stats)
#         self.dt = dt
#
#     def dyn_fcn(self, x, q, pars):
#         if x[4] == 0:
#             # zero yaw rate case
#             f = np.array([
#                 self.dt * x[2] * np.cos(x[3]),
#                 self.dt * x[2] * np.sin(x[3]),
#                 self.dt * q[0],
#                 self.dt * x[3] + 0.5 * self.dt ** 2 * q[1],
#                 self.dt * q[1]
#             ])
#         else:
#             c = x[2] / x[4]
#             f = np.array([
#                 c * (np.sin(x[3] + x[4]*self.dt) - np.sin(x[3])) + 0.5 * self.dt**2 * np.cos(x[3]) * q[0],
#                 c * (-np.cos(x[3] + x[4] * self.dt) + np.cos(x[3])) + 0.5 * self.dt**2 * np.sin(x[3]) * q[0],
#                 self.dt * q[0],
#                 self.dt * x[3] + 0.5 * self.dt**2 * q[1],
#                 self.dt * q[1]
#             ])
#         return x + f
#
#     def dyn_fcn_dx(self, x, q, pars):
#         pass
#
#     def meas_fcn(self, x, r, pars):
#         r = np.sqrt(x[0]**2 + x[1]**2)
#         theta = np.arctan2(x[1], x[0])
#         return np.array([r, theta])
#
#     def meas_fcn_dx(self, x, r, pars):
#         pass
#
#     def par_fcn(self, time):
#         pass


# def ctrv_demo(steps=150, mc_sims=50, outfile=None, show_trajectories=False):
#     # use default filename if unspecified
#     if outfile is None:
#         outfile = 'ctrv_demo_results.dat'
#     outfile = os.path.join('..', outfile)
#
#     if not os.path.exists(outfile):
#         dt = 0.02
#         # TODO: adjust model parameters: decrease initial covariance
#         ssm = ConstantTurnRateAndVelocityGaussSSM(dt)
#         ssm.set_pars('x0_mean', np.array([0, 0, 10, 0*np.pi, 0]))
#         ssm.set_pars('x0_cov', np.diag([1e-6, 1e-6, 1, 0.75*np.pi, 1e-6]))
#
#         x, y = ssm.simulate(steps, mc_sims)
#         if show_trajectories:  # show CTRV trajectories
#             plt.plot(x[0, ...], x[1, ...], alpha=0.15, color='b')
#             plt.show()
#
#         # Initialize filters
#         par_dyn = np.array([[1.0, 1, 1, 1, 1, 1, 1, 1]])
#         par_obs = np.array([[1.0, 0.9, 0.9, 1e4, 1e4, 1e4]])  # np.atleast_2d(np.ones(6))
#         dim_aug = ssm.xD+ssm.qD
#         mul_ut_dyn = np.hstack((np.zeros((dim_aug, 1)), np.eye(dim_aug), 2 * np.eye(dim_aug))).astype(np.int)
#         mul_ut_obs = np.hstack((np.zeros((ssm.xD, 1)), np.eye(ssm.xD), 2 * np.eye(ssm.xD))).astype(np.int)
#         alg = OrderedDict({
#             # GaussianProcessKalman(ssm, par_dyn, par_obs, kernel='rbf', points='ut'),
#             'bsqkf': BayesSardKalman(ssm, par_dyn, par_obs, mul_ut_dyn, mul_ut_obs, points='ut'),
#             'ukf': UnscentedKalman(ssm, beta=0),
#         })
#
#         # kpdyn = np.array([[0.5, 1, 1e3, 1, 1e3, 1e3, 1e3, 1e3],
#         #                   [0.5, 1e3, 1, 1e3, 1, 1e3, 1e3, 1e3],
#         #                   [0.35, 3, 3, 3, 3, 1, 1e3, 1e3],
#         #                   [0.35, 3, 3, 3, 3, 1, 1e3, 1e3],
#         #                   [2.2, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]])
#         # alg['bsqkf'].tf_dyn.model.model_var = multivariate_emv(alg['bsqkf'].tf_dyn, kpdyn, mul_ut_dyn)
#         # kpobs = np.array([[1.0, 1, 1, 1e2, 1e2, 1e2],
#         #                   [1.0, 1.4, 1.4, 1e2, 1e2, 1e2]])
#         # alg['bsqkf'].tf_meas.model.model_var = multivariate_emv(alg['bsqkf'].tf_meas, kpobs, mul_ut_obs)
#         alg['bsqkf'].tf_dyn.model.model_var = np.diag([0.0, 0.0, 0.1, 0.0, 0.0])
#         alg['bsqkf'].tf_meas.model.model_var = np.diag([0.00, 0.00])  # 0 * np.eye(2)
#         print('BSQ EMV\ndyn: {} \nobs: {}'.format(alg['bsqkf'].tf_dyn.model.model_var.diagonal(),
#                                                   alg['bsqkf'].tf_meas.model.model_var.diagonal()))
#
#         num_alg = len(alg)
#         d, steps, mc_sims = x.shape
#         mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
#         for ia, a in enumerate(alg):
#             print('Running {:<5} ... '.format(a.upper()), end='', flush=True)
#             t0 = time.time()
#             for imc in range(mc_sims):
#                 mean[..., imc, ia], cov[..., imc, ia] = alg[a].forward_pass(y[..., imc])
#                 alg[a].reset()
#             print('{:>30}'.format('Done in {:.2f} [sec]'.format(time.time() - t0)))
#
#         # Performance score plots
#         print()
#         print('Computing performance scores ...')
#         error2 = mean.copy()
#         lcr = {
#             'position': np.zeros((steps, mc_sims, num_alg)),
#             'speed': np.zeros((steps, mc_sims, num_alg)),
#             'yaw': np.zeros((steps, mc_sims, num_alg)),
#             'yaw rate': np.zeros((steps, mc_sims, num_alg))
#         }
#         for a in range(num_alg):
#             for k in range(steps):
#                 mse = mse_matrix(x[:, k, :], mean[:, k, :, a])
#                 for imc in range(mc_sims):
#                     error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
#                     lcr['position'][k, imc, a] = log_cred_ratio(x[:2, k, imc], mean[:2, k, imc, a],
#                                                                 cov[:2, :2, k, imc, a], mse[:2, :2])
#                     lcr['speed'][k, imc, a] = log_cred_ratio(x[2, k, imc], mean[2, k, imc, a],
#                                                              cov[2, 2, k, imc, a], mse[2, 2])
#                     lcr['yaw'][k, imc, a] = log_cred_ratio(x[3, k, imc], mean[3, k, imc, a],
#                                                            cov[3, 3, k, imc, a], mse[3, 3])
#                     lcr['yaw rate'][k, imc, a] = log_cred_ratio(x[4, k, imc], mean[4, k, imc, a],
#                                                                 cov[4, 4, k, imc, a], mse[4, 4])
#
#         # Averaged RMSE and Inclination Indicator in time
#         rmse_vs_time = {
#             'position': np.sqrt((error2[:2, ...]).sum(axis=0)).mean(axis=1),
#             'speed': np.sqrt((error2[2, ...])).mean(axis=1),
#             'yaw': np.sqrt((error2[3, ...])).mean(axis=1),
#             'yaw rate': np.sqrt((error2[4, ...])).mean(axis=1),
#         }
#
#         # Save the simulation results into outfile
#         data_dict = {
#             'steps': steps,
#             'dt': dt,
#             'alg_str': list(alg.keys()),
#             'position': {'rmse': rmse_vs_time['position'], 'inc': lcr['position'].mean(axis=1)},
#             'speed': {'rmse': rmse_vs_time['speed'], 'inc': lcr['speed'].mean(axis=1)},
#             'yaw': {'rmse': rmse_vs_time['yaw'], 'inc': lcr['yaw'].mean(axis=1)},
#             'yaw rate': {'rmse': rmse_vs_time['yaw rate'], 'inc': lcr['yaw rate'].mean(axis=1)},
#         }
#         # joblib.dump(data_dict, outfile)
#         print('RMSE')
#         for k in rmse_vs_time.keys():
#             print('{:<10}: {}'.format(k, rmse_vs_time[k].mean(axis=0)))
#         print('INC')
#         for k in lcr.keys():
#             print('{:<10}: {}'.format(k, lcr[k].mean(axis=(0, 1))))
#
#         # Visualize simulation results, plots, tables
#         ctrv_demo_results(data_dict)
#     else:
#         ctrv_demo_results(joblib.load(outfile))


def ctrv_demo_results(data_dict):
    alg_str = data_dict['alg_str']
    # PLOTS
    steps, dt = data_dict['steps'], data_dict['dt']
    state_labels = ['Position', 'Speed', 'Yaw', 'Yaw Rate']
    time_sec = np.linspace(0, steps*dt - dt, steps)

    for state_label in state_labels:
        fig, ax = plt.subplots(2, 1, sharex=True)
        fig.suptitle(state_label)
        for i, f_str in enumerate(alg_str):
            ax[0].plot(time_sec, data_dict[state_label.lower()]['rmse'][:, i], label=f_str.upper(), lw=2)
        ax[0].set_ylabel('RMSE')
        ax[0].legend()
        for i, f_str in enumerate(alg_str):
            ax[1].plot(time_sec, data_dict[state_label.lower()]['inc'][:, i], label=f_str.upper(), lw=2)
        ax[1].add_line(Line2D([0, steps], [0, 0], linewidth=2, color='k', ls='--'))
        ax[1].set_ylabel('INC')
        ax[1].set_xlabel('time [sec]')
    plt.show()


def reentry_analysis_demo(dur=50, mc_sims=10):
    tau = 0.05
    disc_tau = 0.1

    # initial condition statistics
    m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932])
    P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 0])
    init_rv = GaussRV(5, m0, P0)

    # process noise statistics
    noise_rv = GaussRV(3, cov=np.diag([2.4064e-5, 2.4064e-5, 0]))
    sys = ReentryVehicle2DTransition(init_rv, noise_rv)

    # measurement noise statistics
    dim_obs = 2
    meas_noise_rv = GaussRV(dim_obs, cov=np.diag([1e-6, 0.17e-6]))
    obs = Radar2DMeasurement(meas_noise_rv, 5, radar_loc=np.array([6374, 0.0]))

    np.random.seed(0)
    # Generate reference trajectory by SDE simulation
    x = sys.simulate_continuous(duration=dur, dt=tau, mc_sims=mc_sims)
    y = obs.simulate_measurements(x)

    # subsample data for the filter and performance score calculation
    x = x[:, ::2, ...]  # take every second point on the trajectory
    y = y[:, ::2, ...]  # and corresponding measurement

    # Initialize state-space model with mis-specified initial mean
    m0 = np.array([6500.4, 349.14, -1.1093, -6.1967, 0.6932])
    P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
    init_rv = GaussRV(5, m0, P0)
    noise_rv = GaussRV(3, cov=np.diag([2.4064e-5, 2.4064e-5, 1e-6]))
    dyn = ReentryVehicle2DTransition(init_rv, noise_rv, dt=disc_tau)
    # NOTE: higher tau causes higher spread of trajectories at the ends

    # Initialize filters
    par_dyn = np.array([[1.0, 1, 1, 1, 1, 1]])
    par_obs = np.array([[1.0, 0.9, 0.9, 1e4, 1e4, 1e4]])  # np.atleast_2d(np.ones(6))
    mul_ut = np.hstack((np.zeros((dyn.dim_in, 1)), np.eye(dyn.dim_in), 2 * np.eye(dyn.dim_in))).astype(np.int)
    alg = OrderedDict({
        # GaussianProcessKalman(ssm, par_dyn, par_obs, kernel='rbf', points='ut'),
        'bsqkf': BayesSardKalman(dyn, obs, par_dyn, par_obs, mul_ut, mul_ut, points='ut'),
        'ukf': UnscentedKalman(dyn, obs, kappa=0.0, beta=2.0),
    })

    alg['bsqkf'].tf_dyn.model.model_var = np.diag([0.1, 0.1, 0.1, 0.1, 0.015])
    alg['bsqkf'].tf_obs.model.model_var = 0 * np.eye(2)
    print('BSQ EMV\ndyn: {} \nobs: {}'.format(alg['bsqkf'].tf_dyn.model.model_var.diagonal(),
                                              alg['bsqkf'].tf_obs.model.model_var.diagonal()))

    # FILTERING
    num_alg = len(alg)
    d, steps, mc_sims = x.shape
    mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
    mean_x, cov_x = mean.copy(), cov.copy()
    mean_y, cov_y = np.zeros((dim_obs, steps, mc_sims, num_alg)), np.zeros((dim_obs, dim_obs, steps, mc_sims, num_alg))
    cov_yx = np.zeros((dim_obs, d, steps, mc_sims, num_alg))

    # y = np.hstack((np.zeros((dim_obs, 1, mc_sims)), y))
    for ia, a in enumerate(alg):
        print('Running {:<5} ... '.format(a.upper()), end='', flush=True)
        t0 = time.time()
        for imc in range(mc_sims):
            for k in range(steps):

                # TIME UPDATE
                alg[a]._time_update(k)
                # record predictive moments
                mean_x[:, k, imc, ia], cov_x[..., k, imc, ia] = alg[a].x_mean_pr, alg[a].x_cov_pr
                mean_y[:, k, imc, ia], cov_y[..., k, imc, ia] = alg[a].y_mean_pr, alg[a].y_cov_pr
                cov_yx[..., k, imc, ia] = alg[a].xy_cov

                # MEASUREMENT UPDATE
                alg[a]._measurement_update(y[:, k, imc])
                # record filtered moments
                mean[:, k, imc, ia], cov[..., k, imc, ia] = alg[a].x_mean_fi, alg[a].x_cov_fi

            # reset filter storage
            alg[a].reset()
        print('{:>30}'.format('Done in {:.2f} [sec]'.format(time.time() - t0)))

    # dd = {
    #     'x': x,
    #     'mean': mean,
    #     'cov': cov,
    #     'mean_x': mean_x,
    #     'cov_x': cov_x,
    #     'mean_y': mean_y,
    #     'cov_y': cov_y,
    #     'cov_yx': cov_yx
    # }

    # cho_solve(cho_factor(self.y_cov_pr), self.xy_cov).T


if __name__ == '__main__':
    reentry_demo(mc_sims=10)
    # reentry_analysis_demo()

    # dd = joblib.load('../reentry_demo_results.dat')
    # reentry_demo_results(dd)
