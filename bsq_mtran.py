import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from ssmtoybox.mtran import UnscentedTransform, SphericalRadialTransform, MonteCarloTransform
from ssmtoybox.bq.bqmtran import BayesSardTransform, GaussianProcessTransform
from ssmtoybox.utils import symmetrized_kl_divergence


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


def sum_of_squares_demo(plot_fit=False):
    dims = [1, 2, 3, 5, 10, 15, 25]

    sos_mean_data = np.zeros((2, len(dims)))
    sos_var_data = sos_mean_data.copy()
    for d, dim_in in enumerate(dims):
        alpha_ut = np.hstack((np.zeros((dim_in, 1)), np.eye(dim_in), 2*np.eye(dim_in))).astype(np.int)
        kpar = np.array([[1.0] + dim_in*[2.0]])
        tforms = OrderedDict({
            'bsq': BayesSardTransform(dim_in, 1, kpar, alpha_ut, point_str='ut', point_par={'kappa': 0.0}),
            'ut': UnscentedTransform(dim_in, kappa=0.0, beta=0.0)
        })

        # print('EMV (dim = {:d}): {:.2e}'.format(dim_in, tforms['bsq'].model.model_var))

        mean_in = np.zeros((dim_in, ))
        cov_in = np.eye(dim_in)

        for t, tf_key in enumerate(tforms):
            mean_tf, cov_tf, cc = tforms[tf_key].apply(sos, mean_in, cov_in, None)
            sos_mean_data[t, d] = np.asscalar(mean_tf)
            sos_var_data[t, d] = np.asscalar(cov_tf)

    import pandas as pd
    row_labels = [tstr for tstr in tforms.keys()]
    col_labels = [str(d) for d in dims]

    table_mean = pd.DataFrame(sos_mean_data, index=row_labels, columns=col_labels)
    table_var = pd.DataFrame(sos_var_data, index=row_labels, columns=col_labels)

    pd.set_option('precision', 2)
    print(table_mean)
    print(table_var)

    if plot_fit:
        bsqmt = BayesSardTransform(1, 1, np.array([[1.0, 3.0]]), np.array([[0, 1, 2]], dtype=np.int), point_str='ut')
        xtest = np.linspace(-5, 5, 50)
        ytrue = np.zeros((len(xtest)))
        for i in range(len(xtest)):
            ytrue[i] = sos(xtest[i], None)
        y = np.zeros((bsqmt.model.points.shape[1], ))
        for i in range(bsqmt.model.points.shape[1]):
            y[i] = sos(bsqmt.model.points[:, i], None)
        bsqmt.model.plot_model(xtest[None, :], fcn_obs=y, fcn_true=ytrue)


def polar2cartesian(x, pars):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def polar2cartesian_skl_demo():
    dim = 2

    # create spiral in polar domain
    r_spiral = lambda x: 10 * x
    theta_min, theta_max = 0.25 * np.pi, 2.25 * np.pi

    # equidistant points on a spiral
    num_mean = 10
    theta_pt = np.linspace(theta_min, theta_max, num_mean)
    r_pt = r_spiral(theta_pt)

    # setup input moments: means are placed on the points of the spiral
    num_cov = 10  # num_cov covariances are considered for each mean
    r_std = 0.5
    theta_std = np.deg2rad(np.linspace(6, 36, num_cov))
    mean = np.array([r_pt, theta_pt])
    cov = np.zeros((dim, dim, num_cov))
    for i in range(num_cov):
        cov[..., i] = np.diag([r_std**2, theta_std[i]**2])

    # COMPARE moment transforms
    ker_par = np.array([[1.0, 60, 6]])
    mul_ind = np.hstack((np.zeros((dim, 1)), np.eye(dim), 2*np.eye(dim))).astype(np.int)
    tforms = OrderedDict([
        ('bsq-ut', BayesSardTransform(dim, dim, ker_par, mul_ind, point_str='ut', point_par={'kappa': 2, 'alpha': 1})),
        ('gpq-ut', GaussianProcessTransform(dim, dim, ker_par, point_str='ut', point_par={'kappa': 2, 'alpha': 1})),
        ('ut', UnscentedTransform(dim, kappa=2, alpha=1, beta=0)),
    ])
    baseline_mtf = MonteCarloTransform(dim, n=10000)
    num_tforms = len(tforms)

    # initialize storage of SKL scores
    skl_dict = dict([(mt_str, np.zeros((num_mean, num_cov))) for mt_str in tforms.keys()])

    # for each mean
    for i in range(num_mean):

        # for each covariance
        for j in range(num_cov):
            mean_in, cov_in = mean[..., i], cov[..., j]

            # calculate baseline using Monte Carlo
            mean_out_mc, cov_out_mc, cc = baseline_mtf.apply(polar2cartesian, mean_in, cov_in, None)

            # for each moment transform
            for mt_str in tforms.keys():

                # calculate the transformed moments
                mean_out, cov_out, cc = tforms[mt_str].apply(polar2cartesian, mean_in, cov_in, None)

                # compute SKL
                skl_dict[mt_str][i, j] = symmetrized_kl_divergence(mean_out_mc, cov_out_mc, mean_out, cov_out)

    # PLOT the SKL score for each MT and position on the spiral
    plt.style.use('seaborn-deep')
    # printfig = FigurePrint()
    fig = plt.figure()

    # Average over mean indexes
    ax1 = fig.add_subplot(121)
    index = np.arange(num_mean)+1
    for mt_str in tforms.keys():
        ax1.plot(index, skl_dict[mt_str].mean(axis=1), marker='o', label=mt_str.upper())
    ax1.set_xlabel('Position index')
    ax1.set_ylabel('SKL')

    # Average over azimuth variances
    ax2 = fig.add_subplot(122, sharey=ax1)
    for mt_str in tforms.keys():
        ax2.plot(np.rad2deg(theta_std), skl_dict[mt_str].mean(axis=0), marker='o', label=mt_str.upper())
    ax2.set_xlabel('Azimuth STD [$ \circ $]')
    ax2.legend()
    fig.tight_layout(pad=0)
    plt.show()

    # save figure
    # printfig.savefig('polar2cartesian_skl')


if __name__ == '__main__':
    # polar2cartesian_skl_demo()
    sum_of_squares_demo()
