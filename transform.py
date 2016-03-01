from __future__ import division
import numpy as np
from numpy.polynomial.hermite_e import hermegauss, hermeval
from sklearn.utils.extmath import cartesian

class Transform(object):

    def apply(self, f, mean, cov, *args):
        raise NotImplementedError

# TODO: implement classical quadrature-based transforms (spherical-radial, gauss-hermite)
# TODO: implement GPQ+, GPQ+D, GPQ+TD, TPQ+ transforms (adopt from BQ repo)
# TODO: Taylor based transform (EKF); *_eval()'s will need to return Jacobians
# Statistically linearized is pain to use (needs expectations of nonlinearities)


class Linear(Transform):
    # would have to be implemented via first order Taylor, because for linear f(x) = Ax and h(x) = Hx,
    # the Jacobians would be A and H, which mean TaylorFirstOrder is exact inference for linear functions and,
    # in a sense, Kalman filter does not have to be explicitly implemented, because the ExtendedKalman becomes
    # Kalman for linear f() and h().
    pass


class GaussHermite(Transform):

    def __init__(self, degree=3):
        self.degree = degree

    @staticmethod
    def weights(dim, degree):
        pass

    @staticmethod
    def unit_sigma_points(dim, degree):
        pass

    def apply(self, f, mean, cov, *args):
        pass


class SphericalRadial(Transform):

    def __init__(self, dim):
        self.w = self.weights(dim)
        self.W = np.diag(self.w)
        self.unit_sp = self.unit_sigma_points(dim, np.sqrt(dim))

    @staticmethod
    def weights(dim):
        return (1/2*dim) * np.ones(dim)

    @staticmethod
    def unit_sigma_points(dim, c):
        return np.hstack((c*np.eye(dim), -c*np.eye(dim)))

    def apply(self, fcn, mean, cov, *args):
        # form sigma-points from unit sigma-points
        x = mean + np.linalg.cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(fcn, 0, x, *args)
        # output mean
        mean_f = fx.dot(self.w)
        # output covariance
        dfx = fx - mean_f
        cov_f = dfx.dot(self.W).dot(dfx.T)
        # input-output covariance
        cov_fx = dfx.dot(self.W).dot((x - mean).T)
        return mean_f, cov_f, cov_fx


class Unscented(Transform):
    """
    General purpose class implementing Uscented transform.
    """
    def __init__(self, dim, kappa=None, alpha=1, beta=2):
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa
        lam = alpha**2 * (dim + kappa) - dim
        # UT weights
        self.wm, self.wc = self.weights(dim, lam, alpha, beta)
        self.Wm = np.diag(self.wm)
        self.Wc = np.diag(self.wc)
        # UT unit sigma-points1
        self.unit_sp = self.unit_sigma_points(dim, np.sqrt(dim + lam))

    @staticmethod
    def unit_sigma_points(dim, c):
        return np.hstack((np.zeros((dim, 1)), c*np.eye(dim), -c*np.eye(dim)))

    @staticmethod
    def weights(dim, lam, alpha, beta):
        wm = 1.0 / (2.0 * (dim + lam)) * np.ones(2*dim+1)
        wc = wm.copy()
        wm[0] = lam / (dim + lam)
        wc[0] = wm[0] + (1 - alpha**2 + beta)
        return wm, wc

    def apply(self, fcn, mean, cov, *args):  # supply the augmented mean and cov in case noise is non-additive
        # form sigma-points from unit sigma-points
        x = mean + np.linalg.cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(fcn, 0, x, *args)
        # output mean
        mean_f = fx.dot(self.wm)
        # output covariance
        dfx = fx - mean_f
        cov_f = dfx.dot(self.Wc).dot(dfx.T)
        # input-output covariance
        cov_fx = dfx.dot(self.Wc).dot((x - mean).T)
        return mean_f, cov_f, cov_fx
