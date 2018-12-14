from ssmtoybox.utils import *
from paper_code.journal_figure import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ssmtoybox.ssinf import GaussianProcessKalman, UnscentedKalman
from ssmtoybox.ssmod import ReentryVehicle1DTransition, RangeMeasurement, ReentryVehicle2DTransition, Radar2DMeasurement


def reentry_gpq_demo():
    mc_sims = 20
    disc_tau = 0.5  # discretization period

    # ground-truth data generator (true system)
    m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932])
    P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 0])
    x0 = GaussRV(5, m0, P0)
    Q = np.diag([2.4064e-5, 2.4064e-5, 0])
    q = GaussRV(3, cov=Q)
    sys = ReentryVehicle2DTransition(x0, q, dt=disc_tau)

    # radar measurement model
    r = GaussRV(2, cov=np.diag([1e-6, 0.17e-6]))
    radar_x, radar_y = sys.R0, 0
    obs = Radar2DMeasurement(r, 5, radar_loc=np.array([radar_x, radar_y]))

    # Generate reference trajectory by Euler-Maruyama integration
    x = sys.simulate_continuous(duration=200, dt=disc_tau, mc_sims=mc_sims)
    x_ref = x.mean(axis=2)

    # generate radar measurements
    y = obs.simulate_measurements(x)

    # setup SSM for the filter
    m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0])
    P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
    x0 = GaussRV(5, m0, P0)
    q = GaussRV(3, cov=disc_tau*Q)
    dyn = ReentryVehicle2DTransition(x0, q, dt=disc_tau)

    # Initialize filters
    hdyn = np.array([[1.0, 25, 25, 25, 25, 25]])
    hobs = np.array([[1.0, 25, 25, 1e4, 1e4, 1e4]])
    alg = (
        GaussianProcessKalman(dyn, obs, hdyn, hobs, kernel='rbf', points='ut'),
        UnscentedKalman(dyn, obs),
    )

    # Are both filters using the same sigma-points?
    # assert np.array_equal(alg[0].tf_dyn.model.points, alg[1].tf_dyn.unit_sp)

    num_alg = len(alg)
    d, steps, mc_sims = x.shape
    mean, cov = np.zeros((d, steps, mc_sims, num_alg)), np.zeros((d, d, steps, mc_sims, num_alg))
    for imc in range(mc_sims):
        for ia, a in enumerate(alg):
            mean[..., imc, ia], cov[..., imc, ia] = a.forward_pass(y[..., imc])
            a.reset()

    # Plots
    plt.figure()
    g = GridSpec(2, 4)
    plt.subplot(g[:, :2])

    # Earth surface w/ radar position
    t = 0.02 * np.arange(-1, 4, 0.1)
    plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t), color='darkblue', lw=2)
    plt.plot(radar_x, radar_y, 'ko')

    plt.plot(x_ref[0, :], x_ref[1, :], color='r', ls='--')
    # Convert from polar to cartesian
    meas = np.stack((radar_x + y[0, ...] * np.cos(y[1, ...]), radar_y + y[0, ...] * np.sin(y[1, ...])), axis=0)
    for i in range(mc_sims):
        # Vehicle trajectory
        # plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='r', ls='--')

        # Plot measurements
        plt.plot(meas[0, :, i], meas[1, :, i], 'k.', alpha=0.3)

        # Filtered position estimate
        plt.plot(mean[0, 1:, i, 0], mean[1, 1:, i, 0], color='g', alpha=0.3)
        plt.plot(mean[0, 1:, i, 1], mean[1, 1:, i, 1], color='orange', alpha=0.3)

    # Performance score plots
    error2 = mean.copy()
    lcr = np.zeros((steps, mc_sims, num_alg))
    for a in range(num_alg):
        for k in range(steps):
            mse = mse_matrix(x[:4, k, :], mean[:4, k, :, a])
            for imc in range(mc_sims):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                lcr[k, imc, a] = log_cred_ratio(x[:4, k, imc], mean[:4, k, imc, a], cov[:4, :4, k, imc, a], mse)

    # Averaged RMSE and Inclination Indicator in time
    pos_rmse_vs_time = np.sqrt((error2[:2, ...]).sum(axis=0)).mean(axis=1)
    inc_ind_vs_time = lcr.mean(axis=1)

    # Plots
    plt.subplot(g[0, 2:])
    plt.title('RMSE')
    plt.plot(pos_rmse_vs_time[:, 0], label='GPQKF', color='g')
    plt.plot(pos_rmse_vs_time[:, 1], label='UKF', color='r')
    plt.legend()
    plt.subplot(g[1, 2:])
    plt.title('Inclination Indicator $I^2$')
    plt.plot(inc_ind_vs_time[:, 0], label='GPQKF', color='g')
    plt.plot(inc_ind_vs_time[:, 1], label='UKF', color='r')
    plt.legend()
    plt.show()

    print('Average RMSE: {}'.format(pos_rmse_vs_time.mean(axis=0)))
    print('Average I2: {}'.format(inc_ind_vs_time.mean(axis=0)))


def reentry_simple_gpq_demo(dur=30, tau=0.1, mc=100):
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

    # ground-truth data generator (true system)
    m0 = np.array([90, 6, 1.5])
    P0 = np.diag([0.0929, 1.4865, 1e-4])
    x0 = GaussRV(3, m0, P0)
    q = GaussRV(3, cov=np.zeros((3, 3)))
    sys = ReentryVehicle1DTransition(x0, q, dt=tau)

    # Generate reference trajectory
    x = sys.simulate_continuous(dur, mc_sims=mc)
    # pick only non-divergent trajectories
    x = x[..., np.all(x >= 0, axis=(0, 1))]

    # range measurement model
    r = GaussRV(1, cov=np.array([[0.03048 ** 2]]))
    obs = RangeMeasurement(r, 3)
    y = obs.simulate_measurements(x)

    # state-space model used by the filter
    m0 = np.array([90, 6, 1.7])
    P0 = np.diag([0.0929, 1.4865, 1e-4])
    x0 = GaussRV(3, m0, P0)
    q = GaussRV(3, cov=np.zeros((3, 3)))
    dyn = ReentryVehicle1DTransition(x0, q, dt=tau)

    # GPQKF kernel parameters
    kpar_dyn_ut = np.array([[0.5, 10, 10, 10]])
    kpar_obs_ut = np.array([[0.5, 15, 20, 20]])

    # Initialize filters
    alg = (
        GaussianProcessKalman(dyn, obs, kpar_dyn_ut, kpar_obs_ut, kernel='rbf', points='ut'),
        UnscentedKalman(dyn, obs),
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
    filt_labels = ['GPQKF', 'UKF']
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


def reentry_simple_data(dur=30, tau=0.1, mc=100):
    # ground-truth data generator (true system)
    m0 = np.array([90, 6, 1.5])
    P0 = np.diag([0.0929, 1.4865, 1e-4])
    x0 = GaussRV(3, m0, P0)
    q = GaussRV(3, cov=np.zeros((3, 3)))
    sys = ReentryVehicle1DTransition(x0, q, dt=tau)

    # Generate reference trajectory
    x = sys.simulate_continuous(dur, mc_sims=mc)
    # pick only non-divergent trajectories
    x = x[..., np.all(x >= 0, axis=(0, 1))]
    mc = x.shape[2]

    # range measurement model
    r = GaussRV(1, cov=np.array([[0.03048**2]]))
    obs = RangeMeasurement(r, 3)
    y = obs.simulate_measurements(x)

    # state-space model used by the filter
    m0 = np.array([90, 6, 1.7])
    P0 = np.diag([0.0929, 1.4865, 1e-4])
    x0 = GaussRV(3, m0, P0)
    q = GaussRV(3, cov=np.zeros((3, 3)))
    dyn = ReentryVehicle1DTransition(x0, q, dt=tau)

    # GPQKF kernel parameters
    hdyn = np.array([[0.5, 10, 10, 10]])
    hobs = np.array([[0.5, 15, 20, 20]])

    # Initialize filters
    alg = (
        GaussianProcessKalman(dyn, obs, hdyn, hobs, kernel='rbf', points='ut'),
        # CubatureKalman(ssm),
        UnscentedKalman(dyn, obs),
    )

    num_alg = len(alg)
    d, steps, mc = x.shape
    mean, cov = np.zeros((d, steps, mc, num_alg)), np.zeros((d, d, steps, mc, num_alg))
    for imc in range(mc):
        for ia, a in enumerate(alg):
            # Do filtering and reset the filters for each new track
            mean[..., imc, ia], cov[..., imc, ia] = a.forward_pass(y[..., imc])
            a.reset()

    # compute RMSE, Inclination indicator for velocity, position and ballistic parameter
    error2 = mean.copy()
    lcr = mean.copy()

    print("Calculating scores ...")
    for a in range(num_alg):
        for k in range(steps):
            for imc in range(mc):
                error2[:, k, imc, a] = squared_error(x[:, k, imc], mean[:, k, imc, a])
                for dim in range(d):
                    mse = mse_matrix(x[dim, na, k, :], mean[dim, na, k, :, a])
                    lcr[dim, k, imc, a] = log_cred_ratio(x[dim, k, imc], mean[dim, k, imc, a],
                                                         cov[dim, dim, k, imc, a], mse)

    # Averaged position/velocity RMSE and inclination in time
    rmse_vs_time = np.zeros((d, steps, num_alg))
    lcr_vs_time = rmse_vs_time.copy()
    for dim in range(d):
        rmse_vs_time[dim, ...] = np.sqrt(error2[dim, ...]).mean(axis=1)
    lcr_vs_time = lcr.mean(axis=2)

    # time index for plotting
    time = np.linspace(1, dur, x.shape[1])

    # Pack the data into dictionary
    # data_scores = dict([(name, eval(name)) for name in ['time', 'x', 'mean', 'cov', 'rmse_vs_time', 'lcr_vs_time']])
    data_scores = {
        'time': time,
        'x': x,
        'mean': mean,
        'cov': cov,
        'rmse_vs_time': rmse_vs_time,
        'lcr_vs_time': lcr_vs_time
    }
    return data_scores


def reentry_simple_plots(data_scores):

    # unpack from dictionary
    time = data_scores['time']
    rmse_vs_time = data_scores['rmse_vs_time']
    lcr_vs_time = data_scores['lcr_vs_time']
    printfig = FigurePrint()

    d, steps, num_alg = rmse_vs_time.shape
    # RMSE
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=printfig.figsize(h_scale=1.2))
    fig.text(0.00, 0.5, 'RMSE', va='center', rotation='vertical')
    for iax, ax in enumerate(axes):
        for alg in range(num_alg):
            ax.plot(time, rmse_vs_time[iax, :, alg], lw=2)
    axes[-1].set_xlabel('time [s]')
    axes[0].legend(['GPQKF', 'UKF'])
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0.13)  # make room for common Y label
    printfig.savefig("reentry_state_rmse")

    # Inclination
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=printfig.figsize(h_scale=1.2))
    fig.text(0.00, 0.5, r'Inclination $\nu$', va='center', rotation='vertical')
    for iax, ax in enumerate(axes):
        for alg in range(num_alg):
            ax.plot(time, lcr_vs_time[iax, :, alg], lw=2)
    axes[-1].set_xlabel('time [s]')
    axes[0].legend(['GPQKF', 'UKF'], loc='upper left')
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0.13)  # make room for common Y label
    printfig.savefig("reentry_state_inclination")

    # RMSE
    # fig = plt.figure(figsize=figsize())
    # ax1 = fig.add_subplot(311, ylabel='RMSE')
    # ax1.plot(time, pos_rmse_vs_time[:, 0], lw=2, label='GPQKF')
    # ax1.plot(time, pos_rmse_vs_time[:, 1], lw=2, label='UKF')
    # ax1.legend()
    # ax1.tick_params(axis='both', which='both', top='off', right='off', labelright='off', labelbottom='off')
    #
    # ax2 = fig.add_subplot(312, sharex=ax1, ylabel='RMSE')
    # ax2.plot(time, vel_rmse_vs_time[:, 0], lw=2, label='GPQKF')
    # ax2.plot(time, vel_rmse_vs_time[:, 1], lw=2, label='UKF')
    # ax2.tick_params(axis='both', which='both', top='off', right='off', labelright='off', labelbottom='off')
    #
    # ax3 = fig.add_subplot(313, sharex=ax1, xlabel='time [s]', ylabel='RMSE')
    # ax3.plot(time, theta_rmse_vs_time[:, 0], lw=2, label='GPQKF')
    # ax3.plot(time, theta_rmse_vs_time[:, 1], lw=2, label='UKF')
    # ax3.tick_params(axis='both', which='both', top='off', right='off', labelright='off')
    #
    # fig.tight_layout(pad=0, h_pad=0.08)
    # savefig("reentry_state_rmse")

    # inclination indicator
    # fig = plt.figure(figsize=figsize())
    # ax1 = fig.add_subplot(311, ylabel=r'Incl. $\nu$')
    # ax1.plot(time, pos_inc_vs_time[:, 0], lw=2, label='GPQKF')
    # ax1.plot(time, pos_inc_vs_time[:, 1], lw=2, label='UKF')
    # ax1.legend()
    # ax1.tick_params(axis='both', which='both', top='off', right='off', labelright='off', labelbottom='off')
    #
    # ax2 = fig.add_subplot(312, sharex=ax1, ylabel=r'Incl. $\nu$')
    # ax2.plot(time, vel_inc_vs_time[:, 0], lw=2, label='GPQKF')
    # ax2.plot(time, vel_inc_vs_time[:, 1], lw=2, label='UKF')
    # ax2.tick_params(axis='both', which='both', top='off', right='off', labelright='off', labelbottom='off')
    #
    # ax3 = fig.add_subplot(313, sharex=ax1, xlabel='time [s]', ylabel=r'Incl. $\nu$')
    # ax3.plot(time, theta_inc_vs_time[:, 0], lw=2, label='GPQKF')
    # ax3.plot(time, theta_inc_vs_time[:, 1], lw=2, label='UKF')
    # ax3.tick_params(axis='both', which='both', top='off', right='off', labelright='off')
    #
    # fig.tight_layout(pad=0, h_pad=0.08)
    # savefig("reentry_state_inclination")

    # One figure for each RMSE/Inclination plot
    # fig = plt.figure(figsize=figsize(h_scale=0.45))
    # ax = fig.add_subplot(111)
    # ax.plot(time[1:], pos_rmse_vs_time[1:, 0], lw=2, label='GPQKF')
    # ax.plot(time[1:], pos_rmse_vs_time[1:, 1], lw=2, label='UKF')
    # ax.set_xlabel('time [k]')
    # ax.set_ylabel('RMSE')
    # plt.legend()
    # plt.tight_layout(pad=0)
    # savefig("reentry_position_rmse")
    #
    # fig = plt.figure(figsize=figsize(h_scale=0.45))
    # ax = fig.add_subplot(111)
    # ax.plot(time, vel_rmse_vs_time[:, 0], lw=2, label='GPQKF')
    # ax.plot(time, vel_rmse_vs_time[:, 1], lw=2, label='UKF')
    # ax.set_xlabel('time [k]')
    # ax.set_ylabel('RMSE')
    # plt.legend()
    # plt.tight_layout(pad=0)
    # savefig("reentry_velocity_rmse")
    #
    # fig = plt.figure(figsize=figsize(h_scale=0.45))
    # ax = fig.add_subplot(111)
    # ax.plot(time, theta_rmse_vs_time[:, 0], lw=2, label='GPQKF')
    # ax.plot(time, theta_rmse_vs_time[:, 1], lw=2, label='UKF')
    # ax.set_xlabel('time [k]')
    # ax.set_ylabel('RMSE')
    # plt.legend()
    # plt.tight_layout(pad=0)
    # savefig("reentry_theta_rmse")
    #
    # fig = plt.figure(figsize=figsize(h_scale=0.45))
    # ax = fig.add_subplot(111)
    # ax.plot(time[1:], pos_inc_vs_time[1:, 0], lw=2, label='GPQKF')
    # ax.plot(time[1:], pos_inc_vs_time[1:, 1], lw=2, label='UKF')
    # ax.set_xlabel('time [k]')
    # ax.set_ylabel(r'Incl. $\nu$')
    # plt.legend()
    # plt.tight_layout(pad=0)
    # savefig("reentry_position_inclination")
    #
    # fig = plt.figure(figsize=figsize(h_scale=0.45))
    # ax = fig.add_subplot(111)
    # ax.plot(time, vel_inc_vs_time[:, 0], lw=2, label='GPQKF')
    # ax.plot(time, vel_inc_vs_time[:, 1], lw=2, label='UKF')
    # ax.set_xlabel('time [k]')
    # ax.set_ylabel(r'Incl. $\nu$')
    # plt.legend()
    # plt.tight_layout(pad=0)
    # savefig("reentry_velocity_inclination")
    #
    # fig = plt.figure(figsize=figsize(h_scale=0.45))
    # ax = fig.add_subplot(111)
    # ax.plot(time, theta_inc_vs_time[:, 0], lw=2, label='GPQKF')
    # ax.plot(time, theta_inc_vs_time[:, 1], lw=2, label='UKF')
    # ax.set_xlabel('time [k]')
    # ax.set_ylabel(r'Incl. $\nu$')
    # plt.legend()
    # plt.tight_layout(pad=0)
    # savefig("reentry_theta_inclination")


def reentry_simple_trajectory_plot(data_scores):
    time = data_scores['time']
    x = data_scores['x']

    # plt.style.use('seaborn-deep')
    printfig = FigurePrint()
    # PLOTS: Trajectories
    fig = plt.figure()

    # Altitude
    ax1 = fig.add_subplot(211)
    for i in range(10):
        ax1.plot(time, x[0, :, i], alpha=0.35, lw=1, color='k')
    ax1.plot(time, x[0, ...].mean(axis=1), color='r', ls='--')
    ax1.set_ylabel('Altitude [km]')
    ax1.tick_params(labelbottom='off')

    # Velocity
    ax2 = fig.add_subplot(212, sharex=ax1)
    for i in range(10):
        ax2.plot(time, x[1, :, i], alpha=0.35, lw=1, color='k')
    ax2.plot(time, x[1, ...].mean(axis=1), color='r', ls='--')
    ax2.set_ylabel('Velocity [km/s]')
    ax2.set_xlabel('time [s]')

    fig.tight_layout(pad=0)
    printfig.savefig('reentry_pos_vel')


if __name__ == '__main__':
    import pickle
    # get simulation results
    print('Running simulations ...')
    data_dict = reentry_simple_data(mc=100)
    # reentry_simple_gpq_demo(mc=100, dur=30)

    # dump simulated data for fast re-plotting
    print('Pickling data ...')
    with open('reentry_score_data.dat', 'wb') as f:
        pickle.dump(data_dict, f)
        f.close()

    # load pickled data
    print('Unpickling data ...')
    with open('reentry_score_data.dat', 'rb') as f:
        data_dict = pickle.load(f)
        f.close()

    # calculate scores and generate publication ready figures
    reentry_simple_plots(data_dict)
    reentry_simple_trajectory_plot(data_dict)
    reentry_simple_gpq_demo()

    # radar tracking of a reentry vehicle with more complicated dynamics
    # GPQKF fails with posdef: unable to find good kernel parameters
    # reentry_gpq_demo()
