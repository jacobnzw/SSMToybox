from tpq_base import *
from scipy.io import loadmat, savemat
from ssmtoybox.ssinf import StudentProcessKalman, StudentProcessStudent, GaussianProcessKalman, UnscentedKalman, \
    CubatureKalman, FullySymmetricStudent
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement
from ssmtoybox.utils import GaussRV, StudentRV
from ssmtoybox.bq.bqmtran import BQTransform
import matplotlib.pyplot as plt
# from fusion_paper.figprint import *

# def __init__(self):
#     pars = {
#         'x0_mean': np.atleast_1d(0.0),
#         'x0_cov': np.atleast_2d(5.0),
#         'q_mean_0': np.zeros(self.qD),
#         'q_mean_1': np.zeros(self.qD),
#         'q_cov_0': 10 * np.eye(self.qD),
#         'q_cov_1': 100 * np.eye(self.qD),
#         'r_mean_0': np.zeros(self.rD),
#         'r_mean_1': np.zeros(self.rD),
#         'r_cov_0': 0.01 * np.eye(self.rD),
#         'r_cov_1': 1 * np.eye(self.rD),
#     }

#
# def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
#     super(UNGM, self).__init__(**kwargs)
#     kwargs = {
#         'x0_mean': np.atleast_1d(x0_mean),
#         'x0_cov': np.atleast_2d(x0_cov),
#         'x0_dof': 4.0,
#         'q_mean': np.atleast_1d(q_mean),
#         'q_cov': np.atleast_2d(q_cov),
#         'q_dof': 4.0,
#         'r_mean': np.atleast_1d(r_mean),
#         'r_cov': np.atleast_2d(r_cov),
#         'r_dof': 4.0,
#     }


def ungm_demo(steps=250, mc_sims=100):
    # SYSTEM (data generator): dynamics and measurement
    x0_cov = 1.0
    q_cov_0, q_cov_1 = 10.0, 100.0
    r_cov_0, r_cov_1 = 0.01, 1.0
    x0 = GaussRV(1, cov=x0_cov)
    zero_means = (np.zeros((1,)), np.zeros((1,)))
    gm_weights = np.array([0.8, 0.2])
    q_covs = (np.atleast_2d(q_cov_0), np.atleast_2d(q_cov_1))
    q = GaussianMixtureRV(1, zero_means, q_covs, gm_weights)
    dyn = UNGMTransition(x0, q)

    r_covs = (np.atleast_2d(r_cov_0), np.atleast_2d(r_cov_1))
    r = GaussianMixtureRV(1, zero_means, r_covs, gm_weights)
    obs = UNGMMeasurement(r, dyn.dim_state)

    # simulate data
    x = dyn.simulate_discrete(steps, mc_sims)
    z = obs.simulate_measurements(x)

    # STUDENT STATE SPACE MODEL: dynamics and measurement
    nu = 4.0
    x0 = StudentRV(1, scale=(nu-2)/nu*x0_cov, dof=nu)
    q = StudentRV(1, scale=((nu-2)/nu)*q_cov_0, dof=nu)
    dyn = UNGMTransition(x0, q)
    r = StudentRV(1, scale=((nu-2)/nu)*r_cov_0, dof=nu)
    obs = UNGMMeasurement(r, dyn.dim_state)

    # GAUSSIAN SSM for UKF
    x0 = GaussRV(1, cov=x0_cov)
    q = GaussRV(1, cov=q_cov_0)
    dyn_gauss = UNGMTransition(x0, q)
    r = GaussRV(1, cov=r_cov_0)
    obs_gauss = UNGMMeasurement(r, dyn.dim_state)

    # kernel parameters for TPQ and GPQ filters
    # TPQ Student
    # par_dyn_tp = np.array([[1.8, 3.0]])
    # par_obs_tp = np.array([[0.4, 1.0, 1.0]])
    par_dyn_tp = np.array([[3.0, 1.0]])
    par_obs_tp = np.array([[3.0, 3.0]])
    # GPQ Student
    par_dyn_gpqs = par_dyn_tp
    par_obs_gpqs = par_obs_tp
    # GPQ Kalman
    par_dyn_gpqk = np.array([[1.0, 0.5]])
    par_obs_gpqk = np.array([[1.0, 1, 10]])
    # parameters of the point-set
    kappa = 0.0
    par_pt = {'kappa': kappa}

    # FIXME: TPQ filters give too similar results, unlike in the paper, likely because I fiddled with DoF choice in TPQ
    # init filters
    filters = (
        # ExtendedStudent(dyn, obs),
        UnscentedKalman(dyn_gauss, obs_gauss, kappa=kappa),
        # FullySymmetricStudent(dyn, obs, kappa=kappa, dof=3.0),
        FullySymmetricStudent(dyn, obs, kappa=kappa, dof=4.0),
        # FullySymmetricStudent(dyn, obs, kappa=kappa, dof=8.0),
        # FullySymmetricStudent(dyn, obs, kappa=kappa, dof=100.0),
        StudentProcessStudent(dyn, obs, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=3.0, point_par=par_pt),
        # StudentProcessStudent(dyn, obs, par_dyn_tp, par_obs_tp, dof=3.0, dof_tp=4.0, point_par=par_pt),
        StudentProcessStudent(dyn, obs, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=10.0, point_par=par_pt),
        # StudentProcessStudent(dyn, obs, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=100.0, point_par=par_pt),
        StudentProcessStudent(dyn, obs, par_dyn_tp, par_obs_tp, dof=4.0, dof_tp=500.0, point_par=par_pt),
        # GaussianProcessStudent(dyn, obs, par_dyn_gpqs, par_obs_gpqs, dof=10.0, point_hyp=par_pt),
        # StudentProcessKalman(dyn, obs, par_dyn_gpqk, par_obs_gpqk, points='fs', point_hyp=par_pt),
        # GaussianProcessKalman(dyn, obs, par_dyn_tp, par_obs_tp, point_hyp=par_pt),
    )
    itpq = np.argwhere([isinstance(filters[i], StudentProcessStudent) for i in range(len(filters))]).squeeze(axis=1)[0]

    # assign weights approximated by MC with lots of samples
    # very dirty code
    pts = filters[itpq].tf_dyn.model.points
    kern = filters[itpq].tf_dyn.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_dyn, BQTransform):
            f.tf_dyn.wm, f.tf_dyn.Wc, f.tf_dyn.Wcc = wm, wc, wcc
            f.tf_dyn.Q = Q
    pts = filters[itpq].tf_obs.model.points
    kern = filters[itpq].tf_obs.model.kernel
    wm, wc, wcc, Q = rbf_student_mc_weights(pts, kern, int(1e6), 1000)
    for f in filters:
        if isinstance(f.tf_obs, BQTransform):
            f.tf_obs.wm, f.tf_obs.Wc, f.tf_obs.Wcc = wm, wc, wcc
            f.tf_obs.Q = Q

    # print kernel parameters
    import pandas as pd
    parlab = ['alpha'] + ['ell_{}'.format(d + 1) for d in range(x.shape[0])]
    partable = pd.DataFrame(np.vstack((par_dyn_tp, par_obs_tp)), columns=parlab, index=['dyn', 'obs'])
    print()
    print(partable)

    # run all filters
    mf, Pf = run_filters(filters, z)

    # compute average RMSE and INC from filtered trajectories
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
        'steps': steps,
        'mc_sims': mc_sims,
        'par_dyn_tp': par_dyn_tp,
        'par_obs_tp': par_obs_tp,
    }
    savemat('ungm_simdata_{:d}k_{:d}mc'.format(steps, mc_sims), data_dict)

    f_label = [f.__class__.__name__ for f in filters]
    m_label = ['MEAN_RMSE', 'STD(MEAN_RMSE)', 'MEAN_INC', 'STD(MEAN_INC)']
    data = np.array([rmse_avg.mean(axis=0), np.sqrt(var_rmse_avg), lcr_avg.mean(axis=0), np.sqrt(var_lcr_avg)]).T
    table = pd.DataFrame(data, f_label, m_label)
    pd.set_option('display.max_columns', 6)
    print(table)


def ungm_plots_tables(datafile):

    # extract true/filtered state trajectories, measurements and evaluated metrics from *.mat data file
    d = loadmat(datafile)
    x, z, mf, Pf = d['x'], d['z'], d['mf'], d['Pf']
    rmse_avg, lcr_avg = d['rmse_avg'], d['lcr_avg']
    var_rmse_avg, var_lcr_avg = d['var_rmse_avg'].squeeze(), d['var_lcr_avg'].squeeze()
    steps, mc_sims = d['steps'], d['mc_sims']

    # TABLES
    import pandas as pd

    # limit display of decimal places
    pd.set_option('display.precision', 4)

    # filter/metric labels
    f_label = ['UKF', 'SF', r'TPQSF($\nu$=3)', r'TPQSF($\nu$=4)',
               r'TPQSF($\nu$=6)', r'TPQSF($\nu$=8)', r'TPQSF($\nu$=10)', 'GPQSF']
    m_label = ['MEAN_RMSE', 'VAR(MEAN_RMSE)', 'MEAN_INC', 'VAR(MEAN_INC)']

    # form data array, put in DataFrame and print
    data = np.array([rmse_avg.mean(axis=0), var_rmse_avg, lcr_avg.mean(axis=0), var_lcr_avg]).T
    table = pd.DataFrame(data, f_label, m_label)
    print(table)

    # save table to latex
    with open('ungm_rmse_inc.tex', 'w') as f:
        table.to_latex(f)

    # plots
    from figprint import FigurePrint
    fp = FigurePrint()

    # RMSE and INC box plots
    fig, ax = plt.subplots()
    ax.boxplot(rmse_avg)
    ax.set_ylabel('Average RMSE')
    ax.set_ylim(0, 80)
    xtickNames = plt.setp(ax, xticklabels=f_label)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.tight_layout(pad=0.1)
    fp.savefig('ungm_rmse_boxplot')

    fig, ax = plt.subplots()
    ax.boxplot(lcr_avg)
    ax.set_ylabel('Average INC')
    xtickNames = plt.setp(ax, xticklabels=f_label)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.tight_layout(pad=0.1)
    fp.savefig('ungm_inc_boxplot')

    # filtered state and covariance
    # fig, ax = plt.subplots(3, 1, sharex=True)
    # time = np.arange(1, steps + 1)
    # for fi, f in enumerate(filters):
    #     # true state
    #     ax[fi].plot(time, x[0, :, 0], 'r--', alpha=0.5)
    #
    #     # measurements
    #     ax[fi].plot(time, z[0, :, 0], 'k.')
    #
    #     xhat = mf[0, :, 0, fi]
    #     std = np.sqrt(Pf[0, 0, :, 0, fi])
    #     ax[fi].plot(time, xhat, label=f.__class__.__name__)
    #     ax[fi].fill_between(time, xhat - 2 * std, xhat + 2 * std, alpha=0.15)
    #     ax[fi].axis([None, None, -50, 50])
    #     ax[fi].legend()
    # plt.show()
    #
    # # compare posterior variances with outliers
    # plt.figure()
    # plt.plot(time, z[0, :, 0], 'k.')
    # for fi, f in enumerate(filters):
    #     plt.plot(time, 2 * np.sqrt(Pf[0, 0, :, 0, fi]), label=f.__class__.__name__)
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    np.random.seed(42)
    # synthetic_demo(mc_sims=50)
    # lotka_volterra_demo()
    ungm_demo(steps=250, mc_sims=50)
    # ungm_plots_tables('ungm_simdata_250k_500mc')
    # reentry_tracking_demo()
    # constant_velocity_bot_demo()
    # constant_velocity_radar_demo()
    # constant_velocity_radar_plots_tables('cv_radar_simdata_100k_500mc')
    # coordinated_bot_demo(steps=40, mc_sims=100)
    # coordinated_radar_demo(steps=100, mc_sims=100, plots=False)
