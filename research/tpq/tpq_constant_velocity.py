import numpy as np

from tpq_base import *
from scipy.io import savemat, loadmat
from ssmtoybox.ssmod import ConstantVelocity, Radar2DMeasurement
from ssmtoybox.ssinf import StudentProcessKalman, StudentProcessStudent, GaussianProcessKalman, UnscentedKalman, \
    CubatureKalman, FullySymmetricStudent
from ssmtoybox.utils import GaussRV, StudentRV
from ssmtoybox.bq.bqmtran import BQTransform

"""
Tracking of an object behaving according to Constant Velocity model based on radar measurements with glint noise.    
"""


def constant_velocity_radar_demo(steps=100, mc_sims=100):
    print('Constant Velocity Radar Tracking with Glint Noise')
    print('K = {:d}, MC = {:d}'.format(steps, mc_sims))

    # SYSTEM
    m0 = np.array([10000, 300, 1000, -40], dtype=float)
    P0 = np.diag([100**2, 10**2, 100**2, 10**2])
    x0 = GaussRV(4, m0, P0)
    dt = 0.5  # discretization period
    # process noise and noise gain
    Q = np.diag([50, 5])
    G = np.array([[dt ** 2 / 2, 0],
                  [dt, 0],
                  [0, dt ** 2 / 2],
                  [0, dt]])
    q = GaussRV(2, cov=Q)
    dyn = ConstantVelocity(x0, q, dt)
    # measurement noise, where the outliers are modelled by gaussian mixture
    R0 = np.diag([50, 0.4e-6])
    R1 = np.diag([5000, 1.6e-5])  # glint (outlier) RV covariance
    glint_prob = 0.15
    r = GaussianMixtureRV(2, means=(np.zeros((2,)), np.zeros((2,))),
                          covs=(R0, R1), alphas=(1-glint_prob, glint_prob))
    obs = Radar2DMeasurement(r, dyn.dim_state, state_index=[0, 2, 1, 3])

    # SIMULATE DATA
    x = dyn.simulate_discrete(steps, mc_sims)
    z = obs.simulate_measurements(x)

    # STATE SPACE MODEL
    m0 = np.array([10175, 295, 980, -35], dtype=float)
    P0 = np.diag([100 ** 2, 10 ** 2, 100 ** 2, 10 ** 2])
    x0_dof = 1000.0
    x0 = StudentRV(4, m0, ((x0_dof-2)/x0_dof)*P0, x0_dof)
    dt = 0.5  # discretization period
    # process noise and noise gain
    Q = np.diag([50, 5])
    q = StudentRV(2, scale=((x0_dof-2)/x0_dof)*Q, dof=x0_dof)
    dyn = ConstantVelocity(x0, q, dt)

    r_dof = 4.0
    r = StudentRV(2, scale=((r_dof-2)/r_dof)*R0, dof=r_dof)
    obs = Radar2DMeasurement(r, dyn.dim_state)

    # kernel parameters for TPQ and GPQ filters
    # TPQ Student
    par_dyn_tp = np.array([[1, 100, 100, 100, 100]], dtype=float)
    par_obs_tp = np.array([[0.05, 10, 100, 10, 100]], dtype=float)
    # parameters of the point-set
    kappa = 0.0
    par_pt = {'kappa': kappa}

    # print kernel parameters
    import pandas as pd
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(x.shape[0])]
    partable = pd.DataFrame(np.vstack((par_dyn_tp, par_obs_tp)), columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)

    # init filters
    filters = (
        # ExtendedStudent(dyn, obs),
        # UnscentedKalman(dyn, obs, kappa=kappa),
        FullySymmetricStudent(dyn, obs, kappa=kappa, dof=4.0),
        # StudentProcessStudent(dyn, obs, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=4.0, point_par=par_pt),
        # StudentProcessStudent(dyn, obs, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=10.0, point_par=par_pt),
        # StudentProcessStudent(dyn, obs, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=20.0, point_par=par_pt),
        # GaussianProcessKalman(dyn, obs, par_dyn_tp, par_obs_tp, point_hyp=par_pt),
    )
    # itpq = np.argwhere([isinstance(filters[i], StudentProcessStudent) for i in range(len(filters))]).squeeze(axis=1)[0]
    #
    # # assign weights approximated by MC with lots of samples
    # # very dirty code
    # pts = filters[itpq].tf_dyn.model.points
    # kern = filters[itpq].tf_dyn.model.kernel
    # wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(2e6), 1000)
    # for f in filters:
    #     if isinstance(f.tf_dyn, BQTransform):
    #         f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
    #         f.tf_dyn.Q = Q
    # pts = filters[itpq].tf_obs.model.points
    # kern = filters[itpq].tf_obs.model.kernel
    # wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(2e6), 1000)
    # for f in filters:
    #     if isinstance(f.tf_obs, BQTransform):
    #         f.tf_obs.wm, f.tf_obs.Wc, f.tf_obs.Wcc = wm, wc, wcc
    #         f.tf_obs.Q = Q

    # run all filters
    mf, Pf = run_filters(filters, z)

    pos_x, pos_mf, pos_Pf = x[[0, 2], ...], mf[[0, 2], ...], Pf[np.ix_([0, 2], [0, 2])]
    vel_x, vel_mf, vel_Pf = x[[1, 3], ...], mf[[1, 3], ...], Pf[np.ix_([1, 3], [1, 3])]

    # # DEBUG: visual check of the generated data
    # import matplotlib.pyplot as plt
    # def polar2cartesian(a, pars):
    #     return a[0] * np.array([np.cos(a[1]), np.sin(a[1])])
    #
    # z_cart = polar2cartesian(z, None)
    # num_sim = 10
    # num_t = 100
    # plt.figure()
    # plt.plot(x[0, :num_t, :num_sim], x[2, :num_t, :num_sim], alpha=0.5, c='r')
    # plt.plot(z_cart[0, :num_t, :num_sim], z_cart[1, :num_t, :num_sim], alpha=0.2, c='k', marker='o', ls='')
    # plt.plot(pos_mf[0, :num_t, :num_sim, 0], pos_mf[1, :num_t, :num_sim, 0], alpha=0.5, c='g', ls='--')
    #
    # plt.show()

    # evaluate scores
    pos_rmse, pos_lcr = eval_perf_scores(pos_x, pos_mf, pos_Pf)
    vel_rmse, vel_lcr = eval_perf_scores(vel_x, vel_mf, vel_Pf)
    rmse_avg, lcr_avg = eval_perf_scores(x, mf, Pf)

    # variance of average metrics
    from ssmtoybox.utils import bootstrap_var
    var_rmse_avg = np.zeros((len(filters),))
    var_lcr_avg = np.zeros((len(filters),))
    for fi in range(len(filters)):
        var_rmse_avg[fi] = bootstrap_var(rmse_avg[:, fi], int(1e4))
        var_lcr_avg[fi] = bootstrap_var(lcr_avg[:, fi], int(1e4))

    # save trajectories, measurements and metrics to file for later processing (tables, plots)
    data_dict = {
        'x': x,
        'z': z,
        'mf': mf,
        'Pf': Pf,
        'rmse_avg': rmse_avg,
        'lcr_avg': lcr_avg,
        'var_rmse_avg': var_rmse_avg,
        'var_lcr_avg': var_lcr_avg,
        'pos_rmse': pos_rmse,
        'pos_lcr': pos_lcr,
        'vel_rmse': vel_rmse,
        'vel_lcr': vel_lcr,
        'steps': steps,
        'mc_sims': mc_sims,
        'par_dyn_tp': par_dyn_tp,
        'par_obs_tp': par_obs_tp,
        'f_label': ['UKF', 'SF', r'TPQSF($\nu$=20)', 'GPQSF']
    }
    savemat('cv_radar_simdata_{:d}k_{:d}mc'.format(steps, mc_sims), data_dict)

    # print out table
    # mean overall RMSE and INC with bootstrapped variances
    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'STD(MEAN_RMSE)', 'MEAN_INC', 'STD(MEAN_INC)']
    data = np.array([rmse_avg.mean(axis=0), np.sqrt(var_rmse_avg), lcr_avg.mean(axis=0), np.sqrt(var_lcr_avg)]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # mean/max RMSE and INC
    m_label = ['MEAN_RMSE', 'MAX_RMSE', 'MEAN_INC', 'MAX_INC']
    pos_data = np.array([pos_rmse.mean(axis=0), pos_rmse.max(axis=0), pos_lcr.mean(axis=0), pos_lcr.max(axis=0)]).T
    vel_data = np.array([vel_rmse.mean(axis=0), vel_rmse.max(axis=0), vel_lcr.mean(axis=0), vel_lcr.max(axis=0)]).T
    pos_table = pd.DataFrame(pos_data, f_label, m_label)
    pos_table.index.name = 'Position'
    vel_table = pd.DataFrame(vel_data, f_label, m_label)
    vel_table.index.name = 'Velocity'
    print(pos_table)
    print(vel_table)

    # plot metrics
    import matplotlib.pyplot as plt
    time = np.arange(1, steps + 1)
    fig, ax = plt.subplots(2, 1, sharex=True)
    for fi, f in enumerate(filters):
        ax[0].semilogy(time, pos_rmse[..., fi], label=f.__class__.__name__)
        ax[1].semilogy(time, vel_rmse[..., fi], label=f.__class__.__name__)
    plt.legend()
    plt.show()


def constant_velocity_radar_plots_tables(datafile):

    # extract true/filtered state trajectories, measurements and evaluated metrics from *.mat data file
    d = loadmat(datafile)
    # x, z, mf, Pf = d['x'], d['z'], d['mf'], d['Pf']
    rmse_avg, lcr_avg = d['rmse_avg'], d['lcr_avg']
    var_rmse_avg, var_lcr_avg = d['var_rmse_avg'].squeeze(), d['var_lcr_avg'].squeeze()
    pos_rmse, pos_lcr = d['pos_rmse'], d['pos_lcr']
    vel_rmse, vel_lcr = d['vel_rmse'], d['vel_lcr']
    steps, mc_sims = d['steps'], d['mc_sims']

    # TABLES
    import pandas as pd

    # limit display of decimal places
    pd.set_option('display.precision', 4)

    # filter/metric labels
    # f_label = d['f_label']
    f_label = ['UKF', 'SF', 'TPQSF\n' + r'$(\nu_g=4)$', 'TPQSF\n' + r'$(\nu_g=10)$',
               'TPQSF\n' + r'$(\nu_g=20)$', 'GPQSF']
    m_label = ['MEAN_RMSE', 'STD(MEAN_RMSE)', 'MEAN_INC', 'STD(MEAN_INC)']

    # form data array, put in DataFrame and print
    data = np.array([rmse_avg.mean(axis=0), np.sqrt(var_rmse_avg), lcr_avg.mean(axis=0), np.sqrt(var_lcr_avg)]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # save table to latex
    with open('cv_radar_rmse_inc.tex', 'w') as f:
        table.to_latex(f)

    # plots
    import matplotlib.pyplot as plt
    from figprint import FigurePrint
    fp = FigurePrint()

    # position and velocity RMSE plots
    time = np.arange(1, steps+1)
    fig, ax = plt.subplots(2, 1, sharex=True)

    for fi, f in enumerate(f_label):
        ax[0].semilogy(time, pos_rmse[..., fi], label=f)
        ax[1].semilogy(time, vel_rmse[..., fi], label=f)
    ax[0].set_ylabel('Position')
    ax[1].set_ylabel('Velocity')
    ax[1].set_xlabel('time step [k]')
    plt.legend()
    plt.tight_layout(pad=0)
    fp.savefig('cv_radar_rmse_semilogy')

    # RMSE and INC box plots
    fig, ax = plt.subplots()
    ax.boxplot(rmse_avg, showfliers=True)
    ax.set_ylabel('Average RMSE')
    ax.set_ylim(0, 200)
    xtickNames = plt.setp(ax, xticklabels=f_label)
    # plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.tight_layout(pad=0.1)
    fp.savefig('cv_radar_rmse_boxplot')

    fig, ax = plt.subplots()
    ax.boxplot(lcr_avg)
    ax.set_ylabel('Average INC')
    xtickNames = plt.setp(ax, xticklabels=f_label)
    # plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.tight_layout(pad=0.1)
    fp.savefig('cv_radar_inc_boxplot')


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    np.random.seed(42)  # for reproducibility

    constant_velocity_radar_demo(mc_sims=50)
    # constant_velocity_radar_plots_tables('cv_radar_simdata_100k_100mc')
