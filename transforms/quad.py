import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from numpy.polynomial.hermite_e import hermegauss, hermeval
from scipy.special import factorial
from sklearn.utils.extmath import cartesian

from .mtform import MomentTransform, SigmaPointTransform, SigmaPointTruncTransform


# TODO: add higher-order fully symmetric rules from [McNamee, Stenger]
# TODO: Monte Carlo transform discussed in [Gustafsson, 2012] might serve as baseline for all Gaussian filters
#  Gaussian Particle filter [Djuric] appears to be different from MCT


class MonteCarlo(MomentTransform):
    """Monte Carlo transform.

    Serves as baseline for comparing all other moment transforms.
    """

    def __init__(self, dim, n=100):
        n = int(n)
        self.wm, self.Wc = self.weights(n)
        self.unit_sp = self.unit_sigma_points(dim, n)

    def apply(self, f, mean, cov, pars):
        mean = mean[:, na]
        # form sigma-points from unit sigma-points
        x = mean + la.cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(f, 0, x, pars)
        # output mean
        mean_f = (self.wm * fx).sum(axis=1)
        # output covariance
        dfx = fx - mean_f[:, na]
        cov_f = self.Wc * (dfx.dot(dfx.T))
        # input-output covariance
        cov_fx = self.Wc * dfx.dot((x - mean).T)
        return mean_f, cov_f, cov_fx

    @staticmethod
    def weights(n):
        return 1.0 / n, 1.0 / (n - 1)

    @staticmethod
    def unit_sigma_points(dim, n):
        return np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=n).T


class SphericalRadial(SigmaPointTransform):
    # Could be implemented with Unscented with kappa=0, alpha=1, beta=0.
    def __init__(self, dim):
        self.wm = self.weights(dim)
        self.Wc = np.diag(self.wm)
        self.unit_sp = self.unit_sigma_points(dim)

    @staticmethod
    def weights(dim):
        return (1 / (2.0 * dim)) * np.ones(2 * dim)

    @staticmethod
    def unit_sigma_points(dim):
        c = np.sqrt(dim)
        return np.hstack((c * np.eye(dim), -c * np.eye(dim)))


class SphericalRadialTrunc(SigmaPointTruncTransform):
    def __init__(self, dim, dim_eff):
        self.dim, self.dim_eff = dim, dim_eff
        # weights & points for transformed mean and covariance
        self.wm = SphericalRadial.weights(dim_eff)
        self.Wc = np.diag(self.wm)
        self.unit_sp_eff = SphericalRadial.unit_sigma_points(dim_eff)
        # weights & points for input-output covariance
        self.Wcc = np.diag(SphericalRadial.weights(dim))
        self.unit_sp = SphericalRadial.unit_sigma_points(dim)


class Unscented(SigmaPointTransform):
    """
    General purpose class implementing Uscented transform.
    """

    def __init__(self, dim, kappa=None, alpha=1.0, beta=2.0):
        # UT weights
        self.wm, self.wc = self.weights(dim, kappa=kappa, alpha=alpha, beta=beta)
        self.Wm = np.diag(self.wm)
        self.Wc = np.diag(self.wc)
        # UT unit sigma-points
        self.unit_sp = self.unit_sigma_points(dim, kappa=kappa, alpha=alpha)

    @staticmethod
    def unit_sigma_points(dim, kappa=None, alpha=1.0):
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa
        lam = alpha ** 2 * (dim + kappa) - dim
        c = np.sqrt(dim + lam)
        return np.hstack((np.zeros((dim, 1)), c * np.eye(dim), -c * np.eye(dim)))

    @staticmethod
    def weights(dim, kappa=None, alpha=1.0, beta=2.0):
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa
        lam = alpha ** 2 * (dim + kappa) - dim
        wm = 1.0 / (2.0 * (dim + lam)) * np.ones(2 * dim + 1)
        wc = wm.copy()
        wm[0] = lam / (dim + lam)
        wc[0] = wm[0] + (1 - alpha ** 2 + beta)
        return wm, wc


class UnscentedTrunc(SigmaPointTruncTransform):
    def __init__(self, dim, dim_eff, kappa=None, alpha=1.0, beta=2.0):
        self.dim, self.dim_eff = dim, dim_eff
        # weights & points for transformed mean and covariance
        self.wm, wc = Unscented.weights(dim_eff, kappa, alpha, beta)
        self.Wc = np.diag(wc)
        self.unit_sp_eff = Unscented.unit_sigma_points(dim_eff, kappa, alpha)
        # weights & points for input-output covariance
        self.Wcc = np.diag(Unscented.weights(dim, kappa, alpha, beta)[1])
        self.unit_sp = Unscented.unit_sigma_points(dim, kappa, alpha)


class GaussHermite(SigmaPointTransform):
    def __init__(self, dim, degree=3):
        self.degree = degree
        self.wm = self.weights(dim, degree)
        self.Wc = np.diag(self.wm)
        self.unit_sp = self.unit_sigma_points(dim, degree)

    @staticmethod
    def weights(dim, degree=3):
        # 1D sigma-points (x) and weights (w)
        x, w = hermegauss(degree)
        # hermegauss() provides weights that cause posdef errors
        w = factorial(degree) / (degree ** 2 * hermeval(x, [0] * (degree - 1) + [1]) ** 2)
        return np.prod(cartesian([w] * dim), axis=1)

    @staticmethod
    def unit_sigma_points(dim, degree=3):
        # 1D sigma-points (x) and weights (w)
        x, w = hermegauss(degree)
        # nD sigma-points by cartesian product
        return cartesian([x] * dim).T  # column/sigma-point


class GaussHermiteTrunc(SigmaPointTruncTransform):
    def __init__(self, dim, dim_eff, degree=3):
        self.dim, self.dim_eff = dim, dim_eff
        # weights & points for transformed mean and covariance
        self.wm = GaussHermite.weights(dim_eff, degree)
        self.Wc = np.diag(self.wm)
        self.unit_sp_eff = GaussHermite.unit_sigma_points(dim_eff, degree)
        # weights & points for input-output covariance
        self.Wcc = np.diag(GaussHermite.weights(dim, degree))
        self.unit_sp = GaussHermite.unit_sigma_points(dim, degree)
