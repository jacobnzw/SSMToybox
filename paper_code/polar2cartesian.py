# import matplotlib as mpl
# import matplotlib.pyplot as plt
from paper_code.journal_figure import *
from ssmtoybox.bq.bqmtran import GaussianProcessTransform
from ssmtoybox.mtran import MonteCarloTransform, SphericalRadialTransform, GaussHermiteTransform
from ssmtoybox.utils import *
import numpy.linalg as la
from collections import OrderedDict


"""
Gaussian Process Quadrature moment transformation tested on a mapping from polar to cartesian coordinates.
"""


def no_par(f):
    def wrapper(x):
        return f(x, None)
    return wrapper


def polar2cartesian(x, pars):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def cartesian2polar(x, pars):
    r = np.sqrt(x[0]**2 + x[1]**2)
    theta = np.arctan2(x[1], x[0])
    return np.array([r, theta])


def gpq_polar2cartesian_demo():
    dim = 2

    # Initialize transforms
    # high el[0], because the function is linear given x[1]
    kpar = np.array([[1.0, 600, 6]])
    tf_gpq = GaussianProcessTransform(dim, 1, kpar, kern_str='rbf', point_str='sr')
    tf_sr = SphericalRadialTransform(dim)
    tf_mc = MonteCarloTransform(dim, n=1e4)  # 10k samples

    # Input mean and covariance
    mean_in = np.array([1, np.pi / 2])
    cov_in = np.diag([0.05 ** 2, (np.pi / 10) ** 2])
    # mean_in = np.array([10, 0])
    # cov_in = np.diag([0.5**2, (5*np.pi/180)**2])

    # Mapped samples
    x = np.random.multivariate_normal(mean_in, cov_in, size=int(1e3)).T
    fx = np.apply_along_axis(polar2cartesian, 0, x, None)

    # MC transformed moments
    mean_mc, cov_mc, cc_mc = tf_mc.apply(polar2cartesian, mean_in, cov_in, None)
    ellipse_mc = ellipse_points(mean_mc, cov_mc)

    # GPQ transformed moments with ellipse points
    mean_gpq, cov_gpq, cc = tf_gpq.apply(polar2cartesian, mean_in, cov_in, None)
    ellipse_gpq = ellipse_points(mean_gpq, cov_gpq)

    # SR transformed moments with ellipse points
    mean_sr, cov_sr, cc = tf_sr.apply(polar2cartesian, mean_in, cov_in, None)
    ellipse_sr = ellipse_points(mean_sr, cov_sr)

    # Plots
    plt.figure()

    # MC ground truth mean w/ covariance ellipse
    plt.plot(mean_mc[0], mean_mc[1], 'ro', markersize=6, lw=2)
    plt.plot(ellipse_mc[0, :], ellipse_mc[1, :], 'r--', lw=2, label='MC')

    # GPQ transformed mean w/ covariance ellipse
    plt.plot(mean_gpq[0], mean_gpq[1], 'go', markersize=6)
    plt.plot(ellipse_gpq[0, :], ellipse_gpq[1, :], color='g', label='GPQ')

    # SR transformed mean w/ covariance ellipse
    plt.plot(mean_sr[0], mean_sr[1], 'bo', markersize=6)
    plt.plot(ellipse_sr[0, :], ellipse_sr[1, :], color='b', label='SR')

    # Transformed samples of the input random variable
    plt.plot(fx[0, :], fx[1, :], 'k.', alpha=0.15)
    plt.axes().set_aspect('equal')
    plt.legend()
    plt.show()

    np.set_printoptions(precision=2)
    print("GPQ")
    print("Mean weights: {}".format(tf_gpq.wm))
    print("Cov weight matrix eigvals: {}".format(la.eigvals(tf_gpq.Wc)))
    print("Integral variance: {:.2e}".format(tf_gpq.model.integral_variance(None)))
    print("Expected model variance: {:.2e}".format(tf_gpq.model.exp_model_variance(None)))
    print("SKL Score:")
    print("SR: {:.2e}".format(symmetrized_kl_divergence(mean_mc, cov_mc, mean_sr, cov_sr)))
    print("GPQ: {:.2e}".format(symmetrized_kl_divergence(mean_mc, cov_mc, mean_gpq, cov_gpq)))


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

    # COMPARE moment transforms
    ker_par = np.array([[1.0, 60, 6]])
    moment_tforms = OrderedDict([
        ('gpq-sr', GaussianProcessTransform(num_dim, 1, ker_par, kern_str='rbf', point_str='sr')),
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


def polar2cartesian_spiral_demo():
    num_dim = 2

    # create spiral in polar domain
    r_spiral = lambda x: 10 * x
    theta_min, theta_max = 0.25 * np.pi, 2.25 * np.pi
    theta = np.linspace(theta_min, theta_max, 100)
    r = r_spiral(theta)

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
        cov[..., i] = np.diag([r_std ** 2, theta_std[i] ** 2])

    pol_spiral = np.array([r, theta])
    pol_spiral_pt = np.array([r_pt, theta_pt])

    printfig = FigurePrint()
    fig = plt.figure()
    # PLOTS: Input moments in polar coordinates
    # ax = fig.add_subplot(121, projection='polar')
    # # ax.set_aspect('equal')
    #
    # # origin
    # ax.plot(0, 0, 'r+', ms=4)
    #
    # # spiral
    # ax.plot(pol_spiral[1, :], pol_spiral[0, :], color='r', lw=0.5, ls='--', alpha=0.5)
    #
    # # points on a spiral, i.e. input means
    # ax.plot(pol_spiral_pt[1, :], pol_spiral_pt[0, :], 'o', color='k', ms=1)
    #
    # # for every input mean and covariance
    # for i in range(num_mean):
    #     for j in range(num_cov):
    #
    #         # plot covariance ellipse
    #         car_ellipse = ellipse_points(mean[..., i], cov[..., 5])
    #         ax.plot(car_ellipse[1, :], car_ellipse[0, :], color='k', lw=0.5)

    # transform spiral to Cartesian coordinates
    car_spiral = np.apply_along_axis(polar2cartesian, 0, pol_spiral, None)
    car_spiral_pt = np.apply_along_axis(polar2cartesian, 0, pol_spiral_pt, None)

    # PLOTS: Transformed moments in Cartesian coordinates
    ax = fig.add_subplot(111, projection='polar')
    ax.set_aspect('equal')

    # origin
    ax.plot(0, 0, 'r+', ms=10)

    # spiral
    # ax.plot(car_spiral[0, :], car_spiral[1, :], color='r', lw=0.5, ls='--', alpha=0.5)

    # points on a spiral, i.e. input means
    ax.plot(pol_spiral_pt[0, :], pol_spiral_pt[1, :], 'o', color='k', ms=4)
    ax.text(pol_spiral_pt[0, 5]-0.15, pol_spiral_pt[1, 5]+1.25, r'$[r_i, \theta_i]$')
    rgr = [2, 4, 6]
    plt.rgrids(rgr, [str(r) for r in rgr])
    ax.grid(True, linestyle=':', lw=1, alpha=0.5)

    # for every input mean and covariance
    # for i in range(num_mean):
    #     for j in range(num_cov):
    #
    #         # plot covariance ellipse
    #         car_ellipse = np.apply_along_axis(polar2cartesian, 0, ellipse_points(mean[..., i], cov[..., 5]), None)
    #         # car_ellipse = ellipse_points(mean[..., i], cov[..., -1])
    #         ax.plot(car_ellipse[0, :], car_ellipse[1, :], color='k', lw=0.5)

    fig.tight_layout(pad=0.1)

    printfig.savefig('polar2cartesian_spiral')


if __name__ == '__main__':
    polar2cartesian_skl_demo()
    polar2cartesian_spiral_demo()
