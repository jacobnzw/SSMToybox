from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na, linalg as la
from numpy.linalg import cholesky
from numpy.polynomial.hermite_e import hermegauss, hermeval
from scipy.special import factorial
from sklearn.utils.extmath import cartesian


class MomentTransform(metaclass=ABCMeta):
    """Base class for all moment transforms."""
    @abstractmethod
    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        """
        Transform random variable with given mean and covariance.

        Parameters
        ----------
        f : function
            Handle of the nonlinear transforming function acting on the input random variable.

        mean : (dim, ) ndarray
            Mean of the input random variable.

        cov : (dim, dim) ndarray
            Covariance of the input random variable.

        fcn_pars : ndarray
            Parameters of the nonlinear transforming function.

        tf_pars : ndarray, optional
            Parameters of the moment transform.

        Returns
        -------
        mean_f : ndarray
            Mean of the transformed random variable.

        cov_f : ndarray
            Covariance of the transformed random variable.

        cov_fx : ndarray
            Covariance between the transformed random variable and the input random variable.
        """
        pass


class LinearizationTransform(MomentTransform):
    def __init__(self, dim):
        self.dim = dim

    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        mean_f = f(mean, fcn_pars)
        jacobian_f = f(mean, fcn_pars, dx=True)
        # jacobian_f = jacobian_f.reshape(len(mean_f), self.dim)
        cov_fx = jacobian_f.dot(cov)
        cov_f = cov_fx.dot(jacobian_f.T)
        return mean_f, cov_f, cov_fx


class MonteCarloTransform(MomentTransform):
    """Monte Carlo transform.

    Serves as baseline for comparing all other moment transforms.
    """

    def __init__(self, dim, n=100):
        n = int(n)
        self.wm, self.Wc = self.weights(n)
        self.unit_sp = self.unit_sigma_points(dim, n)

    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        mean = mean[:, na]
        # form sigma-points from unit sigma-points
        x = mean + la.cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(f, 0, x, fcn_pars)
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


"""
Sigma-point transforms.
"""


class SigmaPointTransform(MomentTransform):
    """ Base class for all sigma-point moment transforms."""

    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        """
        Transform random variable with given mean and covariance.

        Parameters
        ----------
        f : function
            Handle of the nonlinear transforming function acting on the input random variable.

        mean : (dim, ) ndarray
            Mean of the input random variable.

        cov : (dim, dim) ndarray
            Covariance of the input random variable.

        fcn_pars : ndarray
            Parameters of the nonlinear transforming function.

        tf_pars : ndarray, optional
            Parameters of the moment transform.

        Returns
        -------
        mean_f : ndarray
            Mean of the transformed random variable.

        cov_f : ndarray
            Covariance of the transformed random variable.

        cov_fx : ndarray
            Covariance between the transformed random variable and the input random variable.
        """
        mean = mean[:, na]
        # form sigma-points from unit sigma-points
        x = mean + cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(f, 0, x, fcn_pars)
        # output mean
        mean_f = fx.dot(self.wm)
        # output covariance
        dfx = fx - mean_f[:, na]
        cov_f = dfx.dot(self.Wc).dot(dfx.T)
        # input-output covariance
        cov_fx = dfx.dot(self.Wc).dot((x - mean).T)
        return mean_f, cov_f, cov_fx


class SphericalRadialTransform(SigmaPointTransform):
    """
    Spherical-radial moment transform.

    Notes
    -----
    Equivalent to the Unscented transform with `kappa=0`, `alpha=1`, `beta=0`. Uses `num_points = 2*dim`.

    Parameters
    ----------
    dim : int
        Dimension of the input random variable.
    """

    def __init__(self, dim):
        self.wm = self.weights(dim)
        self.Wc = np.diag(self.wm)
        self.unit_sp = self.unit_sigma_points(dim)

    @staticmethod
    def weights(dim):
        """
        Spherical-radial transform weights.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable.

        Returns
        -------
        w : (num_points, ) ndarray
            Spherical-radial transform weights.
        """
        return (1 / (2.0 * dim)) * np.ones(2 * dim)

    @staticmethod
    def unit_sigma_points(dim):
        """
        Spherical-radial sigma-points.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable.

        Returns
        -------
        : (dim, num_points) ndarray
            Spherical-radial sigma-points.
        """
        c = np.sqrt(dim)
        return np.hstack((c * np.eye(dim), -c * np.eye(dim)))


class UnscentedTransform(SigmaPointTransform):
    """
    Unscented moment transform.

    Parameters
    ----------
    dim : int
        Dimension of the input random variable.

    kappa : float, optional
        Scaling parameter.

    alpha : float, optional
        Parameter affecting covariance.

    beta : float, optional
        Parameter affecting covariance.
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
        """
        Unscented sigma-points.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable.

        kappa : float, optional
            Scaling parameter.

        alpha : float, optional
            Parameter affecting covariance.

        Returns
        -------
        : (dim, num_points) ndarray
            Unscented sigma-points.
        """
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa
        lam = alpha ** 2 * (dim + kappa) - dim
        c = np.sqrt(dim + lam)
        return np.hstack((np.zeros((dim, 1)), c * np.eye(dim), -c * np.eye(dim)))

    @staticmethod
    def weights(dim, kappa=None, alpha=1.0, beta=2.0):
        """
        Unscented transform weights.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable.

        kappa : float, optional
            Scaling parameter.

        alpha : float, optional
            Parameter affecting covariance.

        beta : float, optional
            Parameter affecting covariance.

        Returns
        -------
        w : (num_points, ) ndarray
            Unscented weights for the transformed mean.

        wc : (num_points, ) ndarray
            Unscented weights for the transformed covariance.
        """
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa
        lam = alpha ** 2 * (dim + kappa) - dim
        wm = 1.0 / (2.0 * (dim + lam)) * np.ones(2 * dim + 1)
        wc = wm.copy()
        wm[0] = lam / (dim + lam)
        wc[0] = wm[0] + (1 - alpha ** 2 + beta)
        return wm, wc


class GaussHermiteTransform(SigmaPointTransform):
    """
    Gauss-Hermite moment transform.

    Parameters
    ----------
    dim : int
        Dimension of the input random variable.

    degree : int, optional
        Degree of the integration rule.

    """
    def __init__(self, dim, degree=3):
        self.degree = degree
        self.wm = self.weights(dim, degree)
        self.Wc = np.diag(self.wm)
        self.unit_sp = self.unit_sigma_points(dim, degree)

    @staticmethod
    def weights(dim, degree=3):
        """
        Gauss-Hermite quadrature weights.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable.

        degree : int, optional
            Degree of the integration rule.

        Returns
        -------
        : (num_points, ) ndarray
            GH quadrature weights of given degree.
        """
        # 1D sigma-points (x) and weights (w)
        x, w = hermegauss(degree)
        # hermegauss() provides weights that cause posdef errors
        w = factorial(degree) / (degree ** 2 * hermeval(x, [0] * (degree - 1) + [1]) ** 2)
        return np.prod(cartesian([w] * dim), axis=1)

    @staticmethod
    def unit_sigma_points(dim, degree=3):
        """
        Unit Gauss-Hermite sigma-points of given degree.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable.

        degree : int, optional
            Degree of the integration rule.

        Returns
        -------
        : (dim, num_points) ndarray
            GH sigma-points.
        """
        # 1D sigma-points (x) and weights (w)
        x, w = hermegauss(degree)
        # nD sigma-points by cartesian product
        return cartesian([x] * dim).T  # column/sigma-point


class FullySymmetricStudentTransform(SigmaPointTransform):
    """
    Moment transform for Student's t-distributed random variables based on fully symmetric integration rule from [1]_.

    Parameters
    ----------
    dim : int
        Dimension of the input random variable (Dimension of the integration domain).

    degree : int
        Degree (order) of the quadrature rule.

    kappa : float
        Tuning parameter of the fully-symmetric point set. If `kappa=None`, chooses `kappa = max(3-dim, 0)`.

    dof : float
        Degree of freedom of the input density.

    Notes
    -----
    The weights are coded for rule orders (degrees) 3 and 5 only. The 3rd order weights converge to UT weights for
    :math:`\\nu \\to \\infty`.

    References
    ----------
    .. [1] J. McNamee and F. Stenger, “Construction of fully symmetric numerical integration formulas,”
           Numer. Math., vol. 10, no. 4, pp. 327–344, 1967.
    """

    _supported_degrees_ = [3, 5]

    def __init__(self, dim, degree=3, kappa=None, dof=4):
        # init parameters stored in object variables
        self.degree, self.kappa, self.dof = degree, kappa, dof

        # init weights
        self.wm = self.weights(dim, degree, kappa, dof)
        self.Wc = np.diag(self.wm)

        # init unit sigma-points
        self.unit_sp = self.unit_sigma_points(dim, degree, kappa, dof)

    @staticmethod
    def weights(dim, degree=3, kappa=None, dof=4.0):
        """
        Weights of the fully symmetric rule for Student-t distribution.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable (Dimension of the integration domain).

        degree : int
            Order of the quadrature rule, only `degree=3` or `degree=5` implemented.

        kappa : float
            Tuning parameter controlling spread of points from the center.

        dof : float
            Degree of freedom parameter for the Student distribution.

        Returns
        -------
        : (num_points, ) ndarray
            Quadrature weights.
        """

        if degree not in FullySymmetricStudentTransform._supported_degrees_:
            print("Defaulting to degree 3. Supplied degree {} not supported. Supported degrees: {}", degree,
                  FullySymmetricStudentTransform._supported_degrees_)
            degree = 3

        # use kappa = 3 - dim if kappa not given
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa

        # dof > 2p, where degree = 2p+1
        dof = np.max((dof, degree))

        if degree == 3:  # code for 3rd-order rule

            # number of points for 3rd-order rule
            n = 2*dim + 1

            # weights are parametrized so that ST-3 -> UT-3 for dof -> inf
            w = 1 / (2 * (dim + kappa)) * np.ones(n)
            w[0] = kappa / (dim + kappa)
            return w

        else:  # code for 5th-order rule

            # compute weights in accordance to McNamee & Stenger (1967)
            I0 = 1
            I2 = dof / (dof - 2)
            I22 = dof ** 2 / ((dof - 2) * (dof - 4))
            I4 = 3 * I22

            A0 = I0 - dim * (I2 / I4) ** 2 * (I4 - 0.5 * (dim - 1) * I22)
            A1 = 0.5 * (I2 / I4) ** 2 * (I4 - (dim - 1) * I22)
            A11 = 0.25 * (I2 / I4) ** 2 * I22

            return np.hstack((A0, A1 * np.ones(2*dim), A11 * np.ones(2*dim*(dim-1))))

    @staticmethod
    def unit_sigma_points(dim, degree=3, kappa=None, dof=4.0):
        """
        Fully-symmetric unit sigma-point set.

        Parameters
        ----------
        dim : int
            Dimension of the input random variable (dimension of the integration domain).

        degree : int
            Order of the quadrature rule, only `degree=3` or `degree=5` implemented.

        kappa : float
            Tuning parameter controlling spread of points from the center. If `kappa=None`, chooses
            `kappa = max(3-dim, 0)`.

        dof : float
            Degree of freedom parameter of the input density.

        Returns
        -------
        : (dim, num_pts) ndarray
            Sigma-points.

        """

        if degree not in FullySymmetricStudentTransform._supported_degrees_:
            print("Defaulting to degree 3. Supplied degree {} not supported. Supported degrees: {}", degree,
                  FullySymmetricStudentTransform._supported_degrees_)
            degree = 3

        # use kappa = 3 - dim if kappa not given
        kappa = np.max([3.0 - dim, 0.0]) if kappa is None else kappa

        # dof > 2p, where degree = 2p+1
        dof = np.max((dof, degree))

        if degree == 3:  # code for 3rd order rule

            # pre-computed integrals, check McNamee & Stenger, 1967
            I2 = dof / (dof - 2)
            u = np.sqrt(I2 * (dim + kappa))
            return u * np.hstack((np.zeros((dim, 1)), np.eye(dim), -np.eye(dim)))

        else:  # code for 5th-order rule

            I2 = dof / (dof - 2)
            I4 = 3 * dof ** 2 / ((dof - 2) * (dof - 4))
            u = np.sqrt(I4 / I2)

            sp0 = FullySymmetricStudentTransform.symmetric_set(dim, [])
            sp1 = FullySymmetricStudentTransform.symmetric_set(dim, [u])
            sp2 = FullySymmetricStudentTransform.symmetric_set(dim, [u, u])

            return np.hstack((sp0, sp1, sp2))

    @staticmethod
    def symmetric_set(dim, gen):
        """
        Symmetric point set.

        Parameters
        ----------
        dim : int
            Dimension.

        gen : array_like (1 dimensional)
            Generator.

        Notes
        -----
        Unscented transform points can be recovered by
            a0 = symmetric_set(dim, [])
            a1 = symmetric_set(dim, [1])
            ut = np.hstack((a0, a1))

        Returns
        -------
        : ndarray
            Fully-symmetric point set.
        """

        # if generator has no element
        nzeros = np.zeros((dim, 1))
        if not gen:
            return nzeros

        gen = np.asarray(gen)
        assert gen.ndim == 1, "Generator must be in 1d array_like."

        uind = np.arange(dim)  # indices of variable u for easier indexing
        eps = np.spacing(1.0)  # machine precision for comparisons
        sp = np.empty(shape=(dim, 0))

        for i in range(dim):
            u = nzeros.copy()
            u[i] = gen[0]

            if len(gen) > 1:
                if np.abs(gen[0] - gen[1]) < eps:
                    V = FullySymmetricStudentTransform.symmetric_set(dim - i - 1, gen[1:])
                    for j in range(V.shape[1]):
                        u[i+1:, 0] = V[:, j]
                        sp = np.hstack((sp, u, -u))
                else:
                    V = FullySymmetricStudentTransform.symmetric_set(dim - 1, gen[1:])
                    for j in range(V.shape[1]):
                        u[uind != i, 0] = V[:, j]
                        sp = np.hstack((sp, u, -u))
            else:
                sp = np.hstack((sp, u, -u))

        return sp


"""
Warning: EXPERIMENTAL!

'Truncated' sigma-point transforms.
"""


class TruncatedSigmaPointTransform(SigmaPointTransform):
    """
    Sigma-point transform respecting effective input dimensionality.

    Notes
    -----
    Created mainly for experimental purposes!
    Computing input-output cross-covariance is problematic and needs further thinking.
    """

    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        mean = mean[:, na]

        # consider only effective dimension
        mean_eff = mean[:self.dim_eff]
        cov_eff = cov[:self.dim_eff, :self.dim_eff]

        # form sigma-points from unit sigma-points
        x_eff = mean_eff + cholesky(cov_eff).dot(self.unit_sp_eff)
        x = mean + cholesky(cov).dot(self.unit_sp)

        # push sigma-points through non-linearity
        fx_eff = np.apply_along_axis(f, 0, x_eff, fcn_pars)
        fx = np.apply_along_axis(f, 0, x, fcn_pars)

        # output mean
        mean_f = fx_eff.dot(self.wm)
        # output covariance
        dfx_eff = fx_eff - mean_f[:, na]
        dfx = fx - mean_f[:, na]
        cov_f = dfx_eff.dot(self.Wc).dot(dfx_eff.T)
        # input-output covariance
        cov_fx = dfx.dot(self.Wcc).dot((x - mean).T)
        # cov_fx = None
        return mean_f, cov_f, cov_fx


class TruncatedSphericalRadialTransform(TruncatedSigmaPointTransform):
    def __init__(self, dim, dim_eff):
        self.dim, self.dim_eff = dim, dim_eff
        # weights & points for transformed mean and covariance
        self.wm = SphericalRadialTransform.weights(dim_eff)
        self.Wc = np.diag(self.wm)
        self.unit_sp_eff = SphericalRadialTransform.unit_sigma_points(dim_eff)
        # weights & points for input-output covariance
        self.Wcc = np.diag(SphericalRadialTransform.weights(dim))
        self.unit_sp = SphericalRadialTransform.unit_sigma_points(dim)


class TruncatedUnscentedTransform(TruncatedSigmaPointTransform):
    def __init__(self, dim, dim_eff, kappa=None, alpha=1.0, beta=2.0):
        self.dim, self.dim_eff = dim, dim_eff
        # weights & points for transformed mean and covariance
        self.wm, wc = UnscentedTransform.weights(dim_eff, kappa, alpha, beta)
        self.Wc = np.diag(wc)
        self.unit_sp_eff = UnscentedTransform.unit_sigma_points(dim_eff, kappa, alpha)
        # weights & points for input-output covariance
        self.Wcc = np.diag(UnscentedTransform.weights(dim, kappa, alpha, beta)[1])
        self.unit_sp = UnscentedTransform.unit_sigma_points(dim, kappa, alpha)


class TruncatedGaussHermiteTransform(TruncatedSigmaPointTransform):
    def __init__(self, dim, dim_eff, degree=3):
        self.dim, self.dim_eff = dim, dim_eff
        # weights & points for transformed mean and covariance
        self.wm = GaussHermiteTransform.weights(dim_eff, degree)
        self.Wc = np.diag(self.wm)
        self.unit_sp_eff = GaussHermiteTransform.unit_sigma_points(dim_eff, degree)
        # weights & points for input-output covariance
        self.Wcc = np.diag(GaussHermiteTransform.weights(dim, degree))
        self.unit_sp = GaussHermiteTransform.unit_sigma_points(dim, degree)


"""
Warning: EXPERIMENTAL!

Linearization transform via Gaussian process quadrature with derivative evaluations.
"""


class TaylorGPQDTransform(MomentTransform):
    """
    Transformation equivalent to GPQ+D w/ RBF kernel, single sigma-point at zero and substitution x = m + z in the
    integral. For el --> infinity the transform converges to LinearizationTransform.
    """

    def __init__(self, dim, alpha=1.0, el=1.0):
        self.dim = dim
        self.alpha = alpha
        self.ell = el
        self.Lam = np.diag(el ** 2 * np.ones(dim))
        self.iLam = np.diag(el ** -2 * np.ones(dim))
        self.eye_d = np.eye(dim)
        # lists for logging average model variance and posterior integral variance
        self.mvar_list = []
        self.ivar_list = []

    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        # wm = la.det(cov.dot(self.iLam) + self.eye_d) ** -0.5
        wm = la.det(self.iLam.dot(cov) + self.eye_d) ** -0.5
        fm = f(mean, fcn_pars)
        mean_f = wm * fm
        jacobian_f = f(mean, fcn_pars, dx=True)
        jacobian_f = jacobian_f.reshape(len(mean_f), self.dim)
        # wc = la.det(cov.dot(2 * self.iLam) + self.eye_d) ** -0.5
        wc = la.det(2 * self.iLam.dot(cov) + self.eye_d) ** -0.5
        Wc = 0.5 * self.Lam.dot(la.inv(0.5 * self.Lam + cov)).dot(cov)
        model_var = self.alpha ** 2 - self.alpha ** 2 * wc * (1 + np.trace(Wc.dot(self.iLam)))
        integ_var = self.alpha ** 2 * wc - wm ** 2
        self.mvar_list.append(model_var)
        self.ivar_list.append(integ_var)
        cov_f = wc * (np.outer(fm, fm) + jacobian_f.dot(Wc).dot(jacobian_f.T)) - np.outer(mean_f, mean_f) + model_var
        cov_fx = self.Lam.dot(la.inv(self.Lam + cov)).dot(cov).dot(jacobian_f.T)
        return mean_f, cov_f, cov_fx