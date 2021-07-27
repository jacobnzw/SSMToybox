import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import trange

from matplotlib import cm
from matplotlib.lines import Line2D
from numpy import newaxis as na

from ssmtoybox.mtran import LinearizationTransform, TaylorGPQDTransform, MonteCarloTransform, UnscentedTransform, \
    SphericalRadialTransform
from ssmtoybox.bq.bqmtran import GaussianProcessTransform
from research.gpqd.gpqd_base import GaussianProcessDerTransform
from ssmtoybox.ssmod import UNGMTransition
from ssmtoybox.utils import GaussRV, maha

from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve


def sos(x, pars, dx=False):
    """Sum of squares function.
    Parameters
    ----------
    x : numpy.ndarray 1D-array
    Returns
    -------
    """
    x = np.atleast_1d(x)
    if not dx:
        return np.atleast_1d(np.sum(x ** 2, axis=0))
    else:
        return np.atleast_1d(2 * x).T.flatten()


def toa(x, pars, dx=False):
    """Time of arrival.
    Parameters
    ----------
    x
    Returns
    -------
    """
    x = np.atleast_1d(x)
    if not dx:
        return np.atleast_1d(np.sum(x ** 2, axis=0) ** 0.5)
    else:
        return np.atleast_1d(x * np.sum(x ** 2, axis=0) ** (-0.5)).T.flatten()


def rss(x, pars, dx=False):
    """Received signal strength in dB scale.
    Parameters
    ----------
    x : N-D ndarray
    Returns
    -------
    """
    c = 10
    b = 2
    x = np.atleast_1d(x)
    if not dx:
        return np.atleast_1d(c - b * 10 * np.log10(np.sum(x ** 2, axis=0)))
    else:
        return np.atleast_1d(-b * 20 / (x * np.log(10))).T.flatten()


def doa(x, pars, dx=False):
    """Direction of arrival in 2D.
    Parameters
    ----------
    x : 2-D ndarray
    Returns
    -------
    """
    if not dx:
        return np.atleast_1d(np.arctan2(x[1], x[0]))
    else:
        return np.array([-x[1], x[0]]) / (x[0] ** 2 + x[1] ** 2).T.flatten()


def rdr(x, pars, dx=False):
    """Radar measurements in 2D."""
    if not dx:
        return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])
    else:  # TODO: returned jacobian must be properly flattened, see dyn_eval in ssm
        return np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [np.sin(x[1]), x[0] * np.cos(x[1])]]).T.flatten()


def kl_div(mu0, sig0, mu1, sig1):
    """KL divergence between two Gaussians. """
    k = 1 if np.isscalar(mu0) else mu0.shape[0]
    sig0, sig1 = np.atleast_2d(sig0, sig1)
    dmu = mu1 - mu0
    dmu = np.asarray(dmu)
    det_sig0 = np.linalg.det(sig0)
    det_sig1 = np.linalg.det(sig1)
    inv_sig1 = np.linalg.inv(sig1)
    kl = 0.5 * (np.trace(np.dot(inv_sig1, sig0)) + np.dot(dmu.T, inv_sig1).dot(dmu) + np.log(det_sig1 / det_sig0) - k)
    return np.asscalar(kl)


def kl_div_sym(mu0, sig0, mu1, sig1):
    """Symmetrized KL divergence."""
    return 0.5 * (kl_div(mu0, sig0, mu1, sig1) + kl_div(mu1, sig1, mu0, sig0))


def rel_error(mu_true, mu_approx):
    """Relative error."""
    assert mu_true.shape == mu_approx.shape
    return la.norm((mu_true - mu_approx) / mu_true)


def plot_func(f, d, n=100, xrng=(-3, 3)):
    xmin, xmax = xrng
    x = np.linspace(xmin, xmax, n)
    assert d <= 2, "Dimensions > 2 not supported. d={}".format(d)
    if d > 1:
        X, Y = np.meshgrid(x, x)
        Z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Z[i, j] = f([X[i, j], Y[i, j]], None)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.5, linewidth=0.75)
        ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.viridis)
        ax.contour(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.viridis)
        ax.contour(X, Y, Z, zdir='y', offset=np.max(Y), cmap=cm.viridis)
        plt.show()
    else:
        y = np.zeros(n)
        for i in range(n):
            y[i] = f(x[i], None)
        fig = plt.plot(x, y)
        plt.show()
    return fig


def save_table(table, filename):
    fo = open(filename, 'w')
    table.to_latex(fo)
    fo.close()


def taylor_gpqd_demo(f):
    """Compares performance of GPQ+D-RBF transform w/ finite lengthscale and Linear transform."""
    d = 2  # dimension

    ker_par_gpqd_taylor = np.array([[1.0, 1.0]])  # alpha = 1.0, ell_1 = 1.0
    ker_par_gpq = np.array([[1.0] + d*[1.0]])
    rbf_kernel = {'name': 'rbf', 'params': ker_par_gpq}
    ut_points = {'name': 'ut', 'params': None}
    # function to test on
    f = toa  # sum_of_squares
    transforms = (
        LinearizationTransform(d),
        TaylorGPQDTransform(d, ker_par_gpqd_taylor),
        GaussianProcessTransform(d, 1, rbf_kernel, ut_points),
        GaussianProcessDerTransform(d, 1, rbf_kernel, ut_points),
        UnscentedTransform(d, kappa=0.0),
        # MonteCarlo(d, n=int(1e4)),
    )
    mean = np.array([3, 0])
    cov = np.array([[1, 0],
                    [0, 10]])
    for ti, t in enumerate(transforms):
        mean_f, cov_f, cc = t.apply(f, mean, cov, None)
        print("{}: mean: {}, cov: {}").format(t.__class__.__name__, mean_f, cov_f)


def gpq_int_var_demo():
    """Compares integral variances of GPQ and GPQ+D by plotting."""
    d = 1
    f = UNGMTransition(GaussRV(d), GaussRV(d)).dyn_eval
    mean = np.zeros(d)
    cov = np.eye(d)

    kpar = np.array([[10.0] + d * [0.7]])
    rbf_kernel = {'name': 'rbf', 'params': kpar}
    ut_points = {'name': 'ut', 'params': {'kappa': 0.0}}
    gpq = GaussianProcessTransform(d, 1, rbf_kernel, ut_points)
    gpqd = GaussianProcessDerTransform(d, 1, rbf_kernel, ut_points)
    mct = MonteCarloTransform(d, n=1e4)

    mean_gpq, cov_gpq, cc_gpq = gpq.apply(f, mean, cov, np.atleast_1d(1.0))
    mean_gpqd, cov_gpqd, cc_gpqd = gpqd.apply(f, mean, cov, np.atleast_1d(1.0))
    mean_mc, cov_mc, cc_mc = mct.apply(f, mean, cov, np.atleast_1d(1.0))

    xmin_gpq = norm.ppf(0.0001, loc=mean_gpq, scale=gpq.model.integral_var)
    xmax_gpq = norm.ppf(0.9999, loc=mean_gpq, scale=gpq.model.integral_var)
    xmin_gpqd = norm.ppf(0.0001, loc=mean_gpqd, scale=gpqd.model.integral_var)
    xmax_gpqd = norm.ppf(0.9999, loc=mean_gpqd, scale=gpqd.model.integral_var)
    xgpq = np.linspace(xmin_gpq, xmax_gpq, 500)
    ygpq = norm.pdf(xgpq, loc=mean_gpq, scale=gpq.model.integral_var)
    xgpqd = np.linspace(xmin_gpqd, xmax_gpqd, 500)
    ygpqd = norm.pdf(xgpqd, loc=mean_gpqd, scale=gpqd.model.integral_var)
    plt.figure()
    plt.plot(xgpq, ygpq, lw=2, label='gpq')
    plt.plot(xgpqd, ygpqd, lw=2, label='gpq+d')
    plt.gca().add_line(Line2D([mean_mc, mean_mc], [0, 150], linewidth=2, color='k'))
    plt.legend()
    plt.show()


def gpq_kl_demo():
    """Compares moment transforms in terms of symmetrized KL divergence."""

    # input dimension
    d = 2
    # unit sigma-points
    pts = SphericalRadialTransform.unit_sigma_points(d)
    # derivative mask, which derivatives to use
    dmask = np.arange(pts.shape[1])
    # RBF kernel hyper-parameters
    hyp = {
        'sos': np.array([[10.0] + d*[6.0]]),
        'rss': np.array([[10.0] + d*[0.2]]),
        'toa': np.array([[10.0] + d*[3.0]]),
        'doa': np.array([[1.0] + d*[2.0]]),
        'rdr': np.array([[10.0] + d*[5.0]]),
    }
    # baseline: Monte Carlo transform w/ 20,000 samples
    mc_baseline = MonteCarloTransform(d, n=2e4)
    # tested functions
    # rss has singularity at 0, therefore no derivative at 0
    # toa does not have derivative at 0, for d = 1
    # rss, toa and sos can be tested for all d > 0; physically d=2,3 make sense
    # radar and doa only for d = 2
    test_functions = (
        # sos,
        toa,
        rss,
        doa,
        rdr,
    )

    # fix seed
    np.random.seed(3)

    # moments of the input Gaussian density
    mean = np.zeros(d)
    cov_samples = 100
    # space allocation for KL divergence
    kl_data = np.zeros((3, len(test_functions), cov_samples))
    re_data_mean = np.zeros((3, len(test_functions), cov_samples))
    re_data_cov = np.zeros((3, len(test_functions), cov_samples))

    print('Calculating symmetrized KL-divergences using {:d} covariance samples...'.format(cov_samples))

    rbf_kernel = {'name': 'rbf', 'params': None}
    sr_points = {'name': 'sr', 'params': None}
    for i in trange(cov_samples):
        # random PD matrix
        a = np.random.randn(d, d)
        cov = a.dot(a.T)
        a = np.diag(1.0 / np.sqrt(np.diag(cov)))  # 1 on diagonal
        cov = a.dot(cov).dot(a.T)
        for idf, f in enumerate(test_functions):
            # print "Testing {}".format(f.__name__)
            mean[:d - 1] = 0.2 if f.__name__ in 'rss' else mean[:d - 1]
            mean[:d - 1] = 3.0 if f.__name__ in 'doa rdr' else mean[:d - 1]
            jitter = 1e-8 * np.eye(2) if f.__name__ == 'rdr' else 1e-8 * np.eye(1)
            # baseline moments using Monte Carlo
            mean_mc, cov_mc, cc = mc_baseline.apply(f, mean, cov, None)
            # tested moment transforms
            rbf_kernel['params'] = hyp[f.__name__]  # use kernel parameters for the current non-linearity
            transforms = (
                SphericalRadialTransform(d),
                GaussianProcessTransform(d, 1, rbf_kernel, sr_points),
                GaussianProcessDerTransform(d, 1, rbf_kernel, sr_points, which_der=dmask),
            )
            for idt, t in enumerate(transforms):
                # apply transform
                mean_t, cov_t, cc = t.apply(f, mean, cov, None)
                # calculate KL distance to the baseline moments
                kl_data[idt, idf, i] = kl_div_sym(mean_mc, cov_mc + jitter, mean_t, cov_t + jitter)
                re_data_mean[idt, idf, i] = rel_error(mean_mc, mean_t)
                re_data_cov[idt, idf, i] = rel_error(cov_mc, cov_t)

    # average over MC samples
    kl_data = kl_data.mean(axis=2)
    re_data_mean = re_data_mean.mean(axis=2)
    re_data_cov = re_data_cov.mean(axis=2)

    # put into pandas dataframe for nice printing and latex output
    row_labels = [t.__class__.__name__ for t in transforms]
    col_labels = [f.__name__ for f in test_functions]
    kl_df = pd.DataFrame(kl_data, index=row_labels, columns=col_labels)
    re_mean_df = pd.DataFrame(re_data_mean, index=row_labels, columns=col_labels)
    re_cov_df = pd.DataFrame(re_data_cov, index=row_labels, columns=col_labels)
    return kl_df, re_mean_df, re_cov_df


def gpq_hypers_demo():
    # input dimension, we can only plot d = 1
    d = 1
    # unit sigma-points
    pts = SphericalRadialTransform.unit_sigma_points(d)
    # pts = Unscented.unit_sigma_points(d)
    # pts = GaussHermite.unit_sigma_points(d, degree=5)
    # shift the points away from the singularity
    # pts += 3*np.ones(d)[:, na]
    # derivative mask, which derivatives to use
    dmask = np.arange(pts.shape[1])
    # functions to test
    test_functions = (sos, toa, rss,)
    # RBF kernel hyper-parameters
    hyp = {
        'sos': np.array([[10.0] + d*[6.0]]),
        'rss': np.array([[10.0] + d*[1.0]]),
        'toa': np.array([[10.0] + d*[1.0]]),
    }
    hypd = {
        'sos': np.array([[10.0] + d*[6.0]]),
        'rss': np.array([[10.0] + d*[1.0]]),
        'toa': np.array([[10.0] + d*[1.0]]),
    }
    # GP plots
    # for f in test_functions:
    #     mt = GaussianProcessTransform(d, kern_par=hyp[f.__name__], point_str='sr')
    #     mt.model.plot_model(test_data, fcn_obs, par=None, fcn_true=None, in_dim=0)
    # # GP plots with derivatives
    # for f in test_functions:
    #     mt = GaussianProcessDerTransform(d, kern_par=hypd[f.__name__], point_str='sr', which_der=dmask)
    #     mt.model.plot_model(test_data, fcn_obs, par=None, fcn_true=None, in_dim=0)


def gpq_sos_demo():
    """Sum of squares analytical moments compared with GPQ, GPQ+D and Spherical Radial transforms."""
    # input dimensions
    dims = [1, 5, 10, 25]
    sos_data = np.zeros((6, len(dims)))
    ivar_data = np.zeros((3, len(dims)))
    ivar_data[0, :] = dims
    for di, d in enumerate(dims):
        # input mean and covariance
        mean_in, cov_in = np.zeros(d), np.eye(d)
        # unit sigma-points
        pts = SphericalRadialTransform.unit_sigma_points(d)
        # derivative mask, which derivatives to use
        dmask = np.arange(pts.shape[1])
        # RBF kernel hyper-parameters
        hyp = {
            'gpq': np.array([[1.0] + d*[10.0]]),
            'gpqd': np.array([[1.0] + d*[10.0]]),
        }
        sr_points = {'name': 'sr', 'params': None}
        transforms = (
            SphericalRadialTransform(d),
            GaussianProcessTransform(d, 1, kernel_spec={'name': 'rbf', 'params': hyp['gpq']}, point_spec=sr_points),
            GaussianProcessDerTransform(d, 1, kernel_spec={'name': 'rbf', 'params': hyp['gpqd']}, point_spec=sr_points,
                                        which_der=dmask),
        )
        ivar_data[1, di] = transforms[1].model.integral_var
        ivar_data[2, di] = transforms[2].model.integral_var
        mean_true, cov_true = d, 2 * d
        # print "{:<15}:\t {:.4f} \t{:.4f}".format("True moments", mean_true, cov_true)
        for ti, t in enumerate(transforms):
            m, c, cc = t.apply(sos, mean_in, cov_in, None)
            sos_data[ti, di] = np.asscalar(m)
            sos_data[ti + len(transforms), di] = np.asscalar(c)
            # print "{:<15}:\t {:.4f} \t{:.4f}".format(t.__class__.__name__, np.asscalar(m), np.asscalar(c))
    row_labels = [t.__class__.__name__ for t in transforms]
    col_labels = [str(d) for d in dims]
    sos_table = pd.DataFrame(sos_data, index=row_labels * 2, columns=col_labels)
    ivar_table = pd.DataFrame(ivar_data[1:, :], index=['GPQ', 'GPQ+D'], columns=col_labels)
    return sos_table, ivar_table, ivar_data


def kern_rbf_der(xs, x, alpha=1.0, el=1.0, which_der=None):
    """RBF kernel w/ derivatives."""
    x, xs = np.atleast_2d(x), np.atleast_2d(xs)
    D, N = x.shape
    Ds, Ns = xs.shape
    assert Ds == D
    which_der = np.arange(N) if which_der is None else which_der
    Nd = len(which_der)  # points w/ derivative observations
    # extract hypers
    # alpha, el, jitter = hypers['sig_var'], hypers['lengthscale'], hypers['noise_var']
    iLam = np.diag(el ** -1 * np.ones(D))
    iiLam = np.diag(el ** -2 * np.ones(D))

    x = iLam.dot(x)  # sqrt(Lambda^-1) * X
    xs = iLam.dot(xs)
    Kff = np.exp(2 * np.log(alpha) - 0.5 * maha(xs.T, x.T))  # cov(f(xi), f(xj))
    x = iLam.dot(x)  # Lambda^-1 * X
    xs = iLam.dot(xs)
    XmX = xs[..., na] - x[:, na, :]  # pair-wise differences
    Kfd = np.zeros((Ns, D * Nd))  # cov(f(xi), df(xj))
    Kdd = np.zeros((D * Nd, D * Nd))  # cov(df(xi), df(xj))
    for i in range(Ns):
        for j in range(Nd):
            jstart, jend = j * D, j * D + D
            j_d = which_der[j]
            Kfd[i, jstart:jend] = Kff[i, j_d] * XmX[:, i, j_d]
    for i in range(Nd):
        for j in range(Nd):
            istart, iend = i * D, i * D + D
            jstart, jend = j * D, j * D + D
            i_d, j_d = which_der[i], which_der[j]  # indices of points with derivatives
            Kdd[istart:iend, jstart:jend] = Kff[i_d, j_d] * (iiLam - np.outer(XmX[:, i_d, j_d], XmX[:, i_d, j_d]))
    return Kff, Kfd, Kdd  # np.vstack((np.hstack((Kff, Kfd)), np.hstack((Kfd.T, Kdd))))


def gp_fit_demo(f, pars, xrng=(-1, 1, 50), save_figs=False, alpha=1.0, el=1.0):
    xs = np.linspace(*xrng)  # test set
    fx = np.apply_along_axis(f, 0, xs[na, :], pars).squeeze()
    xtr = np.sqrt(3) * np.array([-1, 1], dtype=float)  # train set
    ytr = np.apply_along_axis(f, 0, xtr[na, :], pars).squeeze()  # function observations + np.random.randn(xtr.shape[0])
    dtr = np.apply_along_axis(f, 0, xtr[na, :], pars, dx=True).squeeze()  # derivative observations
    y = np.hstack((ytr, dtr))
    m, n = len(xs), len(xtr)  # train and test points
    jitter = 1e-8
    # evaluate kernel matrices
    kss, kfd, kdd = kern_rbf_der(xs, xs, alpha=alpha, el=el)
    kff, kfd, kdd = kern_rbf_der(xs, xtr, alpha=alpha, el=el)
    kfy = np.hstack((kff, kfd))
    Kff, Kfd, Kdd = kern_rbf_der(xtr, xtr, alpha=alpha, el=el)
    K = np.vstack((np.hstack((Kff, Kfd)), np.hstack((Kfd.T, Kdd))))
    # GP fit w/ function values only
    kff_iK = cho_solve(cho_factor(Kff + jitter * np.eye(n)), kff.T).T
    gp_mean = kff_iK.dot(ytr)
    gp_var = np.diag(kss - kff_iK.dot(kff.T))
    gp_std = np.sqrt(gp_var)
    # GP fit w/ functionn values and derivatives
    kfy_iK = cho_solve(cho_factor(K + jitter * np.eye(n + n * 1)), kfy.T).T  # kx.dot(inv(K))
    gp_mean_d = kfy_iK.dot(y)
    gp_var_d = np.diag(kss - kfy_iK.dot(kfy.T))
    gp_std_d = np.sqrt(gp_var_d)

    # setup plotting
    fmin, fmax, fp2p = np.min(fx), np.max(fx), np.ptp(fx)
    axis_limits = [-3, 3, fmin - 0.2 * fp2p, fmax + 0.2 * fp2p]
    tick_settings = {'which': 'both', 'bottom': 'off', 'top': 'off', 'left': 'off', 'right': 'off', 'labelleft': 'off',
                     'labelbottom': 'off'}
    # use tex to render text in the figure
    mpl.rc('text', usetex=True)
    # use lmodern font package which is also used in the paper
    mpl.rc('text.latex', preamble=r'\usepackage{lmodern}')
    # sans serif font for figure, size 10pt
    mpl.rc('font', family='sans-serif', size=10)
    plt.style.use('seaborn-paper')
    # set figure width to fit the column width of the article
    pti = 1.0 / 72.0  # 1 inch = 72 points
    fig_width_pt = 244  # obtained from latex using \the\columnwidth
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_w = fig_width_pt * pti * 1.0
    fig_h = fig_w * golden_mean
    plt.figure(figsize=(fig_w, fig_h))

    # # plot ordinary GP regression fit
    # plt.subplot(211)
    # plt.axis(axis_limits)
    # plt.tick_params(**tick_settings)
    # plt.title('GP regression')
    # plt.plot(xs, fx, 'r--', label='true')
    # plt.plot(xtr, ytr, 'ko', ms=8, label='observed fcn values')
    # plt.plot(xs, gp_mean, 'k-', lw=2, label='GP mean')
    # plt.fill_between(xs, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std, color='k', alpha=0.15)
    # # plot GP regression fit w/ derivative observations
    # plt.subplot(212)
    # plt.axis(axis_limits)
    # plt.tick_params(**tick_settings)
    # plt.title('GP regression with gradient observations')
    # plt.plot(xs, fx, 'r--', label='true')
    # plt.plot(xtr, ytr, 'ko', ms=8, label='observed fcn values')
    # plt.plot(xs, gp_mean_d, 'k-', lw=2, label='GP mean')
    # plt.fill_between(xs, gp_mean_d - 2 * gp_std_d, gp_mean_d + 2 * gp_std_d, color='k', alpha=0.15)
    # # plot line segments to indicate derivative observations
    # h = 0.15
    # for i in range(len(dtr)):
    #     x0, x1 = xtr[i] - h, xtr[i] + h
    #     y0 = dtr[i] * (x0 - xtr[i]) + ytr[i]
    #     y1 = dtr[i] * (x1 - xtr[i]) + ytr[i]
    #     plt.gca().add_line(Line2D([x0, x1], [y0, y1], linewidth=6, color='k'))
    # plt.tight_layout()
    # if save_figs:
    #     plt.savefig('{}_gpr_grad_compar.pdf'.format(f.__name__), format='pdf')
    # else:
    #     plt.show()

    # two figure version
    scale = 0.5
    fig_width_pt = 244 / 2
    fig_w = fig_width_pt * pti
    fig_h = fig_w * golden_mean * 1
    # plot ordinary GP regression fit
    plt.figure(figsize=(fig_w, fig_h))
    plt.axis(axis_limits)
    plt.tick_params(**tick_settings)
    plt.plot(xs, fx, 'r--', label='true')
    plt.plot(xtr, ytr, 'ko', ms=8, label='observed fcn values')
    plt.plot(xs, gp_mean, 'k-', lw=2, label='GP mean')
    plt.fill_between(xs, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std, color='k', alpha=0.15)
    plt.tight_layout(pad=0.5)
    if save_figs:
        plt.savefig('{}_gpr_fcn_obs_small.pdf'.format(f.__name__), format='pdf')
    else:
        plt.show()
    # plot GP regression fit w/ derivative observations
    plt.figure(figsize=(fig_w, fig_h))
    plt.axis(axis_limits)
    plt.tick_params(**tick_settings)
    plt.plot(xs, fx, 'r--', label='true')
    plt.plot(xtr, ytr, 'ko', ms=8, label='observed fcn values')
    plt.plot(xs, gp_mean_d, 'k-', lw=2, label='GP mean')
    plt.fill_between(xs, gp_mean_d - 2 * gp_std_d, gp_mean_d + 2 * gp_std_d, color='k', alpha=0.15)
    # plot line segments to indicate derivative observations
    h = 0.15
    for i in range(len(dtr)):
        x0, x1 = xtr[i] - h, xtr[i] + h
        y0 = dtr[i] * (x0 - xtr[i]) + ytr[i]
        y1 = dtr[i] * (x1 - xtr[i]) + ytr[i]
        plt.gca().add_line(Line2D([x0, x1], [y0, y1], linewidth=6, color='k'))
    plt.tight_layout(pad=0.5)
    if save_figs:
        plt.savefig('{}_gpr_grad_obs_small.pdf'.format(f.__name__), format='pdf')
    else:
        plt.show()

        # integral variances
        # d = 1
        # ut_pts = Unscented.unit_sigma_points(d)
        # # f = UNGM().dyn_eval
        # mean = np.zeros(d)
        # cov = np.eye(d)
        # gpq = GPQuad(d, unit_sp=ut_pts, hypers={'sig_var': alpha, 'lengthscale': el * np.ones(d), 'noise_var': 1e-8})
        # gpqd = GPQuadDerRBF(d, unit_sp=ut_pts,
        #                     hypers={'sig_var': alpha, 'lengthscale': el * np.ones(d), 'noise_var': 1e-8},
        #                     which_der=np.arange(ut_pts.shape[1]))
        # mct = MonteCarlo(d, n=2e4)
        # mean_gpq, cov_gpq, cc_gpq = gpq.apply(f, mean, cov, np.atleast_1d(1.0))
        # mean_gpqd, cov_gpqd, cc_gpqd = gpqd.apply(f, mean, cov, np.atleast_1d(1.0))
        # mean_mc, cov_mc, cc_mc = mct.apply(f, mean, cov, np.atleast_1d(1.0))
        #
        # xmin_gpq = norm.ppf(0.0001, loc=mean_gpq, scale=gpq.integral_var)
        # xmax_gpq = norm.ppf(0.9999, loc=mean_gpq, scale=gpq.integral_var)
        # xmin_gpqd = norm.ppf(0.0001, loc=mean_gpqd, scale=gpqd.integral_var)
        # xmax_gpqd = norm.ppf(0.9999, loc=mean_gpqd, scale=gpqd.integral_var)
        # xgpq = np.linspace(xmin_gpq, xmax_gpq, 500)
        # ygpq = norm.pdf(xgpq, loc=mean_gpq, scale=gpq.integral_var)
        # xgpqd = np.linspace(xmin_gpqd, xmax_gpqd, 500)
        # ygpqd = norm.pdf(xgpqd, loc=mean_gpqd, scale=gpqd.integral_var)
        # #
        # plt.figure(figsize=(fig_w, fig_h))
        # plt.axis([np.min([xmin_gpq, xmin_gpqd]), np.max([xmax_gpq, xmax_gpqd]), 0, np.max(ygpqd) + 0.2 * np.ptp(ygpqd)])
        # plt.tick_params(**tick_settings)
        # plt.plot(xgpq, ygpq, 'k-.', lw=2)
        # plt.plot(xgpqd, ygpqd, 'k-', lw=2)
        # plt.gca().add_line(Line2D([mean_mc, mean_mc], [0, 10], color='r', ls='--', lw=2))
        # plt.tight_layout(pad=0.5)
        # if save_figs:
        #     plt.savefig('{}_gpq_int_var.pdf'.format(f.__name__), format='pdf')
        # else:
        #     plt.show()


if __name__ == '__main__':
    # set seed for reproducibility
    np.random.seed(42)

    # # TABLE 1: SUM OF SQUARES: transformed mean and variance, SR vs. GPQ vs. GPQ+D
    print('Table 1: Comparison of transformed mean and variance for increasing dimension D '
          'computed by the SR, GPQ and GPQ+D moment transforms.')
    sos_table, ivar_table, ivar = gpq_sos_demo()
    pd.set_option('display.float_format', '{:.2e}'.format)
    save_table(sos_table, 'sum_of_squares.tex')
    print('Saved in {}'.format('sum_of_squares.tex'))
    print()

    # # TABLE 2: Comparison of variance of the mean integral for GPQ and GPQ+D
    print('Table 2: Comparison of variance of the mean integral for GPQ and GPQ+D.')
    save_table(ivar_table, 'sos_gpq_int_var.tex')
    print('Saved in {}'.format('sos_gpq_int_var.tex'))
    print()

    # FIGURE 2: (a) Approximation used by GPQ, (b) Approximation used by GPQ+D
    print('Figure 2: (a) Approximation used by the GPQ, (b) Approximation used by the GPQ+D.')
    # gp_fit_demo(UNGM().dyn_eval, [1], xrng=(-3, 3, 50), alpha=10.0, el=0.7)
    gp_fit_demo(sos, None, xrng=(-3, 3, 50), alpha=1.0, el=10.0, save_figs=True)
    # gpq_int_var_demo()
    print('Figures saved in {}, {}'.format('sos_gpr_fcn_obs_small.pdf', 'sos_gpr_grad_obs_small.pdf'))
    print()

    # fig = plot_func(rss, 2, n=100)

    # TABLE 4: Comparison of the SR, GPQ and GPQ+D moment transforms in terms of symmetrized KL-divergence.
    print('Table 4: Comparison of the SR, GPQ and GPQ+D moment transforms in terms of symmetrized KL-divergence.')
    kl_tab, re_mean_tab, re_cov_tab = gpq_kl_demo()
    pd.set_option('display.float_format', '{:.2e}'.format)
    print("\nSymmetrized KL-divergence")
    print(kl_tab.T)
    # print("\nRelative error in the mean")
    # print(re_mean_tab)
    # print("\nRelative error in the covariance")
    # print(re_cov_tab)
    with open('kl_div_table.tex', 'w') as fo:
        kl_tab.T.to_latex(fo)
    print('Saved in {}'.format('kl_div_table.tex'))