import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.lines import Line2D
from numpy import newaxis as na

from ssmtoybox.mtran import LinearizationTransform, TaylorGPQDTransform, MonteCarloTransform, UnscentedTransform, \
    GaussHermiteTransform, SphericalRadialTransform
from ssmtoybox.bq.bqmtran import BQTransform, GaussianProcessTransform
from ssmtoybox.bq.bqmod import GaussianProcessModel, Model
from ssmtoybox.bq.bqkern import RBFGauss, Kernel
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement
from ssmtoybox.utils import GaussRV

from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve

# from research.gpqd.bayesquad import GPQuadDerRBF  # TODO: port the GPQ+D for any point-set from GPQuadDerRBF

# Plan to port GPQuadDerRBF GP Quadrature moment transform with derivative observations
# =====================================================================================
# Create the following subclasses in this file, because trying to incorporate MTs with derivatives into the SSMToybox
# would likely open a can of worms!!!
#
# Create BQTransformDer(BQTransform)
#   - overwrite the _apply() mechanism: add evaluations of the function derivatives
#   - the _mean, _covariance, _cross_covariance() mechanisms stay in place
#   - the main issue is implementation of self.model.bq_weights()
#
# Create GaussianProcessDer(GaussianProcess)
#   - overwrite/hijack the bq_weights() and extend with weights for derivatives
#
# Create RBFGaussDer(RBFGauss)
#   - extend with the corresponding kernel expectations
#   - expectations can be found in Aalto Notes or the bayesquad.py


class GaussianProcessDerTransform(BQTransform):

    def __init__(self, dim_in, dim_out, kern_par,
                       point_str='ut', point_par=None, estimate_par=False, which_der=None):
        self.model = GaussianProcessDerModel(dim_in, kern_par, point_str, point_par, estimate_par, which_der)
        self.I_out = np.eye(dim_out)  # pre-allocation for later computations
        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self.weights(kern_par)

    def _fcn_eval(self, fcn, x, fcn_par):
        """
        Evaluations of the integrand, which can comprise function observations as well as derivative observations.

        Parameters
        ----------
        fcn : func
            Integrand as a function handle, which is expected to behave certain way.

        x : ndarray
            Argument (input) of the integrand.

        fcn_par :
            Parameters of the integrand.

        Returns
        -------
        : ndarray
            Function evaluations of shape (out_dim, num_pts).

        Notes
        -----
        Methods in derived subclasses decides whether to return derivatives also
        """
        # should return as many columns as output dims, one column includes function and derivative evaluations
        # for every sigma-point, thus it is (n + n*d,); n = # sigma-points, d = sigma-point dimensionality
        # returned array should be (n + n*d, e); e = output dimensionality
        # evaluate function at sigmas (e, n)
        fx = np.apply_along_axis(fcn, 0, x, fcn_par)
        # Jacobians evaluated only at sigmas specified by which_der array (e * d, n)
        dfx = np.apply_along_axis(fcn, 0, x[:, self.model.which_der], fcn_par, dx=True)
        # stack function values and derivative values into one column
        return np.vstack((fx.T, dfx.T.reshape(self.model.dim_in * len(self.model.which_der), -1))).T


class GaussianProcessDerModel(GaussianProcessModel):
    """Gaussian Process Model with Derivative Observations"""

    _supported_kernels_ = ['rbf-d']

    def __init__(self, dim, kern_par, point_str, point_par=None, estimate_par=False, which_der=None):
        super(GaussianProcessDerModel, self).__init__(dim, kern_par, 'rbf-d', point_str, point_par, estimate_par)
        self.kernel = RBFGaussDer(dim, kern_par)
        # assume derivatives evaluated at all sigmas if unspecified
        self.which_der = which_der if which_der is not None else np.arange(self.num_pts)

    def bq_weights(self, par, *args):
        par = self.kernel.get_parameters(par)
        x = self.points

        # inverse kernel matrix
        iK = self.kernel.eval_inv_dot(par, x, scaling=False)

        # kernel expectations
        q = self.kernel.exp_x_kx(par, x)
        Q = self.kernel.exp_x_kxkx(par, par, x)
        R = self.kernel.exp_x_xkx(par, x)

        # derivative kernel expectations
        qd = self.kernel.exp_x_dkx(par, x, which_der=self.which_der)
        Qfd = self.kernel.exp_x_kxdkx(par, x)
        Qdd = self.kernel.exp_x_dkxdkx(par, x)
        Rd = self.kernel.exp_x_xdkx(par, x)

        # form the "joint" (function value and derivative) kernel expectations
        q_tilde = np.hstack((q.T, qd.T.ravel()))
        Q_tilde = np.vstack((np.hstack((Q, Qfd)), np.hstack((Qfd.T, Qdd))))
        R_tilde = np.hstack((R, Rd))

        # BQ weights in terms of kernel expectations
        w_m = q_tilde.dot(iK)
        w_c = iK.dot(Q_tilde).dot(iK)
        w_cc = R_tilde.dot(iK)

        # save the kernel expectations for later
        self.q, self.Q, self.iK = q_tilde, Q_tilde, iK
        # expected model variance
        self.model_var = self.kernel.exp_x_kxx(par) * (1 - np.trace(Q_tilde.dot(iK)))
        # integral variance
        self.integral_var = self.kernel.exp_xy_kxy(par) - q_tilde.T.dot(iK).dot(q_tilde)

        # covariance weights should be symmetric
        if not np.array_equal(w_c, w_c.T):
            w_c = 0.5 * (w_c + w_c.T)

        return w_m, w_c, w_cc, self.model_var, self.integral_var

    def exp_model_variance(self, par, *args):
        iK = self.kernel.eval_inv_dot(par, self.points)

        Q = self.kernel.exp_x_kxkx(par, par, self.points)
        Qfd = self.kernel.exp_x_kxdkx(par, par, self.points)
        Qdd = self.kernel.exp_x_dkxdkx(par, par, self.points)
        Q_tilde = np.vstack((np.hstack((Q, Qfd)), np.hstack((Qfd.T, Qdd))))

        return self.kernel.exp_x_kxx(par) * (1 - np.trace(Q_tilde.dot(iK)))

    def integral_variance(self, par, *args):
        par = self.kernel.get_parameters(par)  # if par None returns default kernel parameters

        q = self.kernel.exp_x_kx(par, self.points)
        qd = self.kernel.exp_x_dkx(par, self.points)
        q_tilde = np.hstack((q.T, qd.T.ravel()))

        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
        kbar = self.kernel.exp_xy_kxy(par)
        return kbar - q_tilde.T.dot(iK).dot(q_tilde)


class RBFGaussDer(RBFGauss):

    def __init__(self, dim, par, jitter=1e-8):
        super(RBFGaussDer, self).__init__(dim, par, jitter)

    def eval(self, par, x1, x2=None, diag=False, scaling=True, which_der=None):

        if x2 is None:
            x2 = x1.copy()

        alpha, sqrt_inv_lam = RBFGauss._unpack_parameters(par)
        alpha = 1.0 if not scaling else alpha

        x1 = sqrt_inv_lam.dot(x1)  # sqrt(Lam^-1) * x
        x2 = sqrt_inv_lam.dot(x2)
        if diag:  # only diagonal of kernel matrix
            assert x1.shape == x2.shape
            dx = x1 - x2
            Kff = np.exp(2 * np.log(alpha) - 0.5 * np.sum(dx * dx, axis=0))
        else:
            Kff = np.exp(2 * np.log(alpha) - 0.5 * maha(x1.T, x2.T))

        x1, x2 = np.atleast_2d(x1), np.atleast_2d(x2)
        D, N = x1.shape
        Ds, Ns = x2.shape
        assert Ds == D
        which_der = np.arange(N) if which_der is None else which_der
        Nd = len(which_der)  # points w/ derivative observations
        # iLam = np.diag(el ** -1 * np.ones(D))  # sqrt(Lam^-1)
        # iiLam = np.diag(el ** -2 * np.ones(D))  # Lam^-1

        # x1 = iLam.dot(x1)  # sqrt(Lambda^-1) * X
        # x2 = iLam.dot(x2)
        # Kff = np.exp(2 * np.log(alpha) - 0.5 * maha(x2.T, x1.T))  # cov(f(xi), f(xj))
        x1 = sqrt_inv_lam.dot(x1)  # Lambda^-1 * X
        x2 = sqrt_inv_lam.dot(x2)
        inv_lam = sqrt_inv_lam ** 2
        XmX = x2[..., na] - x1[:, na, :]  # pair-wise differences

        # TODO: benchmark vs. np.kron(), replace with np.kron() if possible
        Kfd = np.zeros((Ns, D * Nd))  # cov(f(xi), df(xj))
        for i in range(Ns):
            for j in range(Nd):
                jstart, jend = j * D, j * D + D
                j_d = which_der[j]
                Kfd[i, jstart:jend] = Kff[i, j_d] * XmX[:, i, j_d]

        Kdd = np.zeros((D * Nd, D * Nd))  # cov(df(xi), df(xj))
        for i in range(Nd):
            for j in range(Nd):
                istart, iend = i * D, i * D + D
                jstart, jend = j * D, j * D + D
                i_d, j_d = which_der[i], which_der[j]  # indices of points with derivatives
                Kdd[istart:iend, jstart:jend] = Kff[i_d, j_d] * (inv_lam - np.outer(XmX[:, i_d, j_d], XmX[:, i_d, j_d]))
        if Ns == N:
            return np.vstack((np.hstack((Kff, Kfd)), np.hstack((Kfd.T, Kdd))))
        else:
            return np.hstack((Kff, Kfd))

    def eval_inv_dot(self, par, x, b=None, scaling=True, which_der=None):
        """
        Compute the product of kernel matrix inverse and a vector `b`.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        x : ndarray
            Data set.

        b : None or ndarray, optional
            If `None`, inverse kernel matrix is computed (i.e. `b=np.eye(N)`).

        scaling : bool, optional
            Use scaling parameter of the kernel matrix.

        which_der : ndarray
            Indicates for which points are the derivatives available.

        Returns
        -------
        : (N, N) ndarray
            Product of kernel matrix inverse and vector `b`.
        """
        # if b=None returns inverse of K
        dim, num_pts = x.shape
        which_der = np.arange(num_pts) if which_der is None else which_der
        num_der = len(which_der)  # number of points with derivatives
        K = self.eval(par, x, scaling=scaling, which_der=which_der)
        return self._cho_inv(K + self.jitter * np.eye(num_pts + num_der*dim), b)

    def eval_chol(self, par, x, scaling=True, which_der=None):
        """
        Compute of Cholesky factor of the kernel matrix.

        Parameters
        ----------
        par : (dim+1, ) ndarray
            Kernel parameters.

        x : (dim, N) ndarray
            Data set.

        scaling : bool, optional
            Use scaling parameter of the kernel.

        which_der : ndarray
            Indicates for which points are the derivatives available.

        Returns
        -------
        : (N, N) ndarray
            Cholesky factor of the kernel matrix.
        """
        dim, num_pts = x.shape
        which_der = np.arange(num_pts) if which_der is None else which_der
        num_der = len(which_der)  # number of points with derivatives
        K = self.eval(par, x, scaling=scaling, which_der=which_der)
        return la.cholesky(K + self.jitter * np.eye(num_pts + num_der*dim))

    def exp_x_dkx(self, par, x, scaling=False, which_der=None):
        """Expectation E_x[k_fd(x, x_n)]"""

        dim, num_pts = x.shape
        alpha, sqrt_inv_lam = RBFGauss._unpack_parameters(par)
        # alpha = 1.0 if not scaling else alpha
        inv_lam = sqrt_inv_lam ** 2
        lam = np.diag(inv_lam.diagonal() ** -1)
        which_der = np.arange(num_pts) if which_der is None else which_der

        q = self.exp_x_kx(par, x, scaling)  # kernel mean E_x[k_ff(x, x_n)]

        eye_d = np.eye(dim)
        Sig_q = cho_solve(cho_factor(inv_lam + eye_d), eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        mu_q = inv_lam.dot(eta)  # (D,N)
        r = q[na, which_der] * inv_lam.dot(mu_q[:, which_der] - x[:, which_der])  # -t.dot(iLam) * q  # (D, N)

        return r.T.ravel()  # (1, n_der*D)

    def exp_x_xdkx(self, par, x, scaling=False, which_der=None):
        """Expectation E_x[x k_fd(x, x_m)]"""
        dim, num_pts = x.shape
        which_der = np.arange(num_pts) if which_der is None else which_der
        num_der = len(which_der)
        _, sqrt_inv_lam = RBFGauss._unpack_parameters(par)

        inv_lam = sqrt_inv_lam ** 2
        eye_d = np.eye(dim)

        q = self.exp_x_kx(par, x, scaling)
        Sig_q = cho_solve(cho_factor(inv_lam + eye_d), eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        mu_q = inv_lam.dot(eta)  # (D,N)
        r = q[na, which_der] * inv_lam.dot(mu_q[:, which_der] - x[:, which_der])  # -t.dot(iLam) * q  # (D, N)

        #  quantities for cross-covariance "weights"
        iLamSig = inv_lam.dot(Sig_q)  # (D,D)
        r_tilde = np.empty((dim, num_der * dim))
        for i in range(num_der):
            i_d = which_der[i]
            r_tilde[:, i * dim:i * dim + dim] = q[i_d] * iLamSig + np.outer(mu_q[:, i_d], r[:, i].T)

        return r_tilde  # (dim, num_pts*dim)

    def exp_x_kxdkx(self, par, x, scaling=False, which_der=None):
        """Expectation E_x[k_ff(x_n, x) k_fd(x, x_m)]"""
        dim, num_pts = x.shape
        which_der = np.arange(num_pts) if which_der is None else which_der
        num_der = len(which_der)

        _, sqrt_inv_lam = RBFGauss._unpack_parameters(par)
        inv_lam = sqrt_inv_lam ** 2
        lam = np.diag(inv_lam.diagonal() ** -1)
        eye_d = np.eye(dim)

        # quantities for covariance weights
        Sig_q = cho_solve(cho_factor(inv_lam + eye_d), eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        inn = inv_lam.dot(x)  # inp / el[:, na]**2
        Q = self.exp_x_kxkx(par, par, x, scaling)  # (N,N)

        cho_LamSig = cho_factor(lam + Sig_q)
        eta_tilde = inv_lam.dot(cho_solve(cho_LamSig, eta))  # Lambda^-1(Lambda+Sig_q)^-1*eta
        mu_Q = eta_tilde[..., na] + eta_tilde[:, na, :]  # (D,N_der,N) pairwise sum of pre-multiplied eta's

        E_dfff = np.empty((num_der * dim, num_pts))
        for i in range(num_der):
            for j in range(num_pts):
                istart, iend = i * dim, i * dim + dim
                i_d = which_der[i]
                E_dfff[istart:iend, j] = Q[i_d, j] * (mu_Q[:, i_d, j] - inn[:, i_d])

        return E_dfff.T  # (num_pts, num_der*dim)

    def exp_x_dkxdkx(self, par, x, scaling=False, which_der=None):
        """Expectation E_x[k_df(x_n, x) k_fd(x, x_m)]"""
        dim, num_pts = x.shape
        which_der = np.arange(num_pts) if which_der is None else which_der
        num_der = len(which_der)

        _, sqrt_inv_lam = RBFGauss._unpack_parameters(par)
        inv_lam = sqrt_inv_lam ** 2
        lam = np.diag(inv_lam.diagonal() ** -1)
        eye_d = np.eye(dim)

        # quantities for covariance weights
        Sig_q = cho_solve(cho_factor(inv_lam + eye_d), eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        inn = inv_lam.dot(x)  # inp / el[:, na]**2
        Q = self.exp_x_kxkx(par, par, x, scaling)  # (N,N)

        cho_LamSig = cho_factor(lam + Sig_q)
        Sig_Q = cho_solve(cho_LamSig, Sig_q).dot(inv_lam)  # (D,D) Lambda^-1 (Lambda*(Lambda+Sig_q)^-1*Sig_q) Lambda^-1
        eta_tilde = inv_lam.dot(cho_solve(cho_LamSig, eta))  # Lambda^-1(Lambda+Sig_q)^-1*eta
        mu_Q = eta_tilde[..., na] + eta_tilde[:, na, :]  # (D,N_der,N) pairwise sum of pre-multiplied eta's

        E_dffd = np.empty((num_der * dim, num_der * dim))
        for i in range(num_der):
            for j in range(num_der):
                istart, iend = i * dim, i * dim + dim
                jstart, jend = j * dim, j * dim + dim
                i_d, j_d = which_der[i], which_der[j]
                T = np.outer((inn[:, i_d] - mu_Q[:, i_d, j_d]), (inn[:, j_d] - mu_Q[:, i_d, j_d]).T) + Sig_Q
                E_dffd[istart:iend, jstart:jend] = Q[i_d, j_d] * T

        return E_dffd  # (num_der*dim, num_der*dim)


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
    ut_pts = UnscentedTransform.unit_sigma_points(d)
    gh_pts = GaussHermiteTransform.unit_sigma_points(d, 5)
    # function to test on
    # f = toa  # sum_of_squares
    transforms = (
        LinearizationTransform(d),
        TaylorGPQDTransform(d, alpha=1.0, el=1.0),
        GaussianProcessTransform(d, unit_sp=ut_pts, hypers={'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8}),
        # GPQuadDerRBF(d, unit_sp=ut_pts, hypers={'sig_var': 1.0, 'lengthscale': 1.0 * np.ones(d), 'noise_var': 1e-8},
        #              which_der=np.arange(ut_pts.shape[1])),
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
    gpq = GaussianProcessTransform(d, 1, kern_par=kpar, kern_str='rbf', point_str='ut', point_par={'kappa': 0.0})
    gpqd = GaussianProcessDerTransform(d, 1, kern_par=kpar, point_str='ut', point_par={'kappa': 0.0})
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
    for i in range(cov_samples):
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
            transforms = (
                SphericalRadialTransform(d),
                GaussianProcessTransform(d, kern_par=hyp[f.__name__], point_str='sr'),
                GaussianProcessDerTransform(d, kern_par=hyp[f.__name__], point_str='sr', which_der=dmask),
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
        transforms = (
            SphericalRadialTransform(d),
            GaussianProcessTransform(d, 1, kern_par=hyp['gpq'], point_str='sr'),
            GaussianProcessDerTransform(d, 1, kern_par=hyp['gpqd'], point_str='sr', which_der=dmask),
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


def maha(x, y, V=None):
    """Pair-wise Mahalanobis distance of rows of x and y with given weight matrix V.
    :param x: (n, d) matrix of row vectors
    :param y: (n, d) matrix of row vectors
    :param V: weight matrix (d, d), if V=None, V=eye(d) is used
    :return:
    """
    if V is None:
        V = np.eye(x.shape[1])
    x2V = np.sum(x.dot(V) * x, 1)
    y2V = np.sum(y.dot(V) * y, 1)
    return (x2V[:, na] + y2V[:, na].T) - 2 * x.dot(V).dot(y.T)


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
    mpl.rc('text.latex', preamble=[r'\usepackage{lmodern}'])
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