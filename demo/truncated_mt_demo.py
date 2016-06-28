from transforms.quad import SphericalRadial, SphericalRadialTrunc, MonteCarlo
import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
import matplotlib.pyplot as plt


def ellipse_points(x0, P):
    # x0 center, SPD matrix
    w, v = la.eig(P)
    theta = np.linspace(0, 2 * np.pi)
    t = np.asarray((np.cos(theta), np.sin(theta)))
    return x0[:, na] + np.dot(v, np.sqrt(w[:, na]) * t)


def cartesian2polar(x, pars, dx=False):
    return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2), np.arctan2(x[1], x[0])])


def polar2cartesian(x, pars, dx=False):
    return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])


def spherical_radial_demo():
    dim = [2, 5, 10, 15]
    d_eff = 2
    # compare truncated SR and SR on cartesian2polar transform for increasing state dimensions
    # truncated SR is aware of the effective dimension, so we expect it to be closer to the true covariance
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
        t = SphericalRadialTrunc(d, d_eff)
        s = SphericalRadial(d)
        # input mean ana covariance
        mean, cov = np.zeros(d), np.eye(d)
        mean[:d_eff], cov[:d_eff, :d_eff] = mean_eff, cov_eff
        M[:, i, 0], C[..., i, 0], cc = t.apply(f, mean, cov, None)
        M[:, i, 1], C[..., i, 1], cc = s.apply(f, mean, cov, None)
        X[..., i, 0] = ellipse_points(M[:, i, 0], C[..., i, 0])
        X[..., i, 1] = ellipse_points(M[:, i, 1], C[..., i, 1])

    plt.figure()
    # transformed samples
    plt.plot(fx[0, :], fx[1, :], 'k.', alpha=0.15)
    # MC mean and covariance ground truth
    plt.plot(M_mc[0], M_mc[1], 'ro', markersize=6, lw=2)
    plt.plot(X_mc[0, :], X_mc[1, :], 'r--', lw=2, label='MC')
    # SR and SR-T mean and covariance for various state dimensions
    for i, d in enumerate(dim):
        plt.plot(M[0, i, 0], M[1, i, 0], 'b+', markersize=10, lw=2)
        plt.plot(X[0, :, i, 0], X[1, :, i, 0], color='b', label='SR-T (d={})'.format(d))
        plt.plot(M[0, i, 0], M[1, i, 0], 'go', markersize=6)
        plt.plot(X[0, :, i, 1], X[1, :, i, 1], color='g', label='SR (d={})'.format(d))
    plt.axes().set_aspect('equal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    spherical_radial_demo()
