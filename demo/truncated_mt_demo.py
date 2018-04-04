from mtran import MonteCarlo, Unscented, UnscentedTrunc
import numpy as np
import matplotlib.pyplot as plt
from mtran import MomentTransform
from utils import ellipse_points


def cartesian2polar(x, pars, dx=False):
    return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan2(x[1], x[0])])


def polar2cartesian(x, pars, dx=False):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def mt_trunc_demo(mt_trunc, mt, dim=None, **kwargs):
    """
    Comparison of truncated MT and vanilla MT on polar2cartesian transform for increasing state dimensions. The
    truncated MT is aware of the effective dimension, so we expect it to be closer to the true covariance

    Observation: Output covariance of the Truncated UT stays closer to the MC baseline than the covariance
    produced by the vanilla UT.

    There's some merit to the idea, but the problem is with computing the input-output cross-covariance.


    Parameters
    ----------
    mt_trunc
    mt
    dim
    kwargs

    Returns
    -------

    """

    assert issubclass(mt_trunc, MomentTransform) and issubclass(mt, MomentTransform)

    # state dimensions and effective dimension
    dim = [2, 3, 4, 5] if dim is None else dim
    d_eff = 2

    # observation model
    f = polar2cartesian

    # use MC transform with lots of samples to compute the true transformed moments
    mean_eff, cov_eff = np.array([1, np.pi / 2]), np.diag([0.05 ** 2, (np.pi / 10) ** 2])
    tmc = MonteCarlo(d_eff, n=1e4)
    M_mc, C_mc, cc_mc = tmc.apply(f, mean_eff, cov_eff, None)

    # transformed samples
    x = np.random.multivariate_normal(mean_eff, cov_eff, size=int(1e3)).T
    fx = np.apply_along_axis(f, 0, x, None)
    X_mc = ellipse_points(M_mc, C_mc)

    M = np.zeros((2, len(dim), 2))
    C = np.zeros((2, 2, len(dim), 2))
    X = np.zeros((2, 50, len(dim), 2))
    for i, d in enumerate(dim):
        t = mt_trunc(d, d_eff, **kwargs)
        s = mt(d, **kwargs)

        # input mean and covariance
        mean, cov = np.zeros(d), np.eye(d)
        mean[:d_eff], cov[:d_eff, :d_eff] = mean_eff, cov_eff

        # transfored moments (w/o cross-covariance)
        M[:, i, 0], C[..., i, 0], cc = t.apply(f, mean, cov, None)
        M[:, i, 1], C[..., i, 1], cc = s.apply(f, mean, cov, None)

        # points on the ellipse defined by the transformed mean and covariance for plotting
        X[..., i, 0] = ellipse_points(M[:, i, 0], C[..., i, 0])
        X[..., i, 1] = ellipse_points(M[:, i, 1], C[..., i, 1])

    # PLOTS: transformed samples, MC mean and covariance ground truth
    plt.figure()
    plt.plot(fx[0, :], fx[1, :], 'k.', alpha=0.15)
    plt.plot(M_mc[0], M_mc[1], 'ro', markersize=6, lw=2)
    plt.plot(X_mc[0, :], X_mc[1, :], 'r--', lw=2, label='MC')

    # SR and SR-T mean and covariance for various state dimensions
    # TODO: it's more effective to plot SKL between the transformed and baseline covariances.
    for i, d in enumerate(dim):
        plt.plot(M[0, i, 0], M[1, i, 0], 'b+', markersize=10, lw=2)
        plt.plot(X[0, :, i, 0], X[1, :, i, 0], color='b', label='mt-trunc (d={})'.format(d))
    for i, d in enumerate(dim):
        plt.plot(M[0, i, 0], M[1, i, 0], 'go', markersize=6)
        plt.plot(X[0, :, i, 1], X[1, :, i, 1], color='g', label='mt (d={})'.format(d))
    plt.axes().set_aspect('equal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    mt_trunc_demo(UnscentedTrunc, Unscented, [2, 5, 10, 15], kappa=0)
