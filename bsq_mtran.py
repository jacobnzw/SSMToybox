import os
import numpy as np
from collections import OrderedDict

from ssmtoybox.mtran import UnscentedTransform, SphericalRadialTransform, MonteCarloTransform
from ssmtoybox.bq.bqmtran import BayesSardTransform, GaussianProcessTransform
from ssmtoybox.utils import symmetrized_kl_divergence


def polar2cartesian(x, pars):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def polar2cartesian_skl_demo():
    num_dim = 2

    # create spiral in polar domain
    r_spiral = lambda x: 10 * x
    theta_min, theta_max = 0.25 * np.pi, 2.25 * np.pi

    # equidistant points on a spiral
    num_mean = 10
    theta_pt = np.linspace(theta_min, theta_max, num_mean)
    r_pt = r_spiral(theta_pt)

    # samples from normal RVs centered on the points of the spiral
    mean = np.array([r_pt, theta_pt])
    r_std = 0.5

    # multiple azimuth covariances in increasing order
    num_cov = 10
    theta_std = np.deg2rad(np.linspace(6, 36, num_cov))
    cov = np.zeros((num_dim, num_dim, num_cov))
    for i in range(num_cov):
        cov[..., i] = np.diag([r_std**2, theta_std[i]**2])

    # COMPARE moment transforms # TODO: polar2cartesian transformation, UT vs. BSQ (vs. GPQ)
    ker_par = np.array([[1.0, 60, 6]])
    moment_tforms = OrderedDict([
        ('gpq-sr', GaussianProcessTransform(num_dim, ker_par, kernel='rbf', points='sr')),
        ('sr', SphericalRadialTransform(num_dim)),
    ])
    baseline_mtf = MonteCarloTransform(num_dim, n=10000)
    num_tforms = len(moment_tforms)

    # initialize storage of SKL scores
    skl_dict = dict([(mt_str, np.zeros((num_mean, num_cov))) for mt_str in moment_tforms.keys()])

    # for each mean
    for i in range(num_mean):

        # for each covariance
        for j in range(num_cov):
            mean_in, cov_in = mean[..., i], cov[..., j]

            # calculate baseline using Monte Carlo
            mean_out_mc, cov_out_mc, cc = baseline_mtf.apply(polar2cartesian, mean_in, cov_in, None)

            # for each MT
            for mt_str in moment_tforms.keys():

                # calculate the transformed moments
                mean_out, cov_out, cc = moment_tforms[mt_str].apply(polar2cartesian, mean_in, cov_in, None)

                # compute SKL
                skl_dict[mt_str][i, j] = symmetrized_kl_divergence(mean_out_mc, cov_out_mc, mean_out, cov_out)

    # PLOT the SKL score for each MT and position on the spiral
    plt.style.use('seaborn-deep')
    printfig = FigurePrint()
    fig = plt.figure()

    # Average over mean indexes
    ax1 = fig.add_subplot(121)
    index = np.arange(num_mean)+1
    for mt_str in moment_tforms.keys():
        ax1.plot(index, skl_dict[mt_str].mean(axis=1), marker='o', label=mt_str.upper())
    ax1.set_xlabel('Position index')
    ax1.set_ylabel('SKL')

    # Average over azimuth variances
    ax2 = fig.add_subplot(122, sharey=ax1)
    for mt_str in moment_tforms.keys():
        ax2.plot(np.rad2deg(theta_std), skl_dict[mt_str].mean(axis=0), marker='o', label=mt_str.upper())
    ax2.set_xlabel('Azimuth STD [$ \circ $]')
    ax2.legend()
    fig.tight_layout(pad=0)

    # save figure
    printfig.savefig('polar2cartesian_skl')

