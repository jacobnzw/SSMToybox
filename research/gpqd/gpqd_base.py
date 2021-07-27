import numpy as np
from numpy import newaxis as na, linalg as la
from scipy.linalg import cho_solve, cho_factor

from ssmtoybox.bq.bqkern import RBFGauss
from ssmtoybox.bq.bqmod import GaussianProcessModel
from ssmtoybox.bq.bqmtran import BQTransform
from ssmtoybox.utils import maha


class GaussianProcessDerTransform(BQTransform):

    def __init__(self, dim_in, dim_out, kernel_spec=None, point_spec=None, estimate_par=False, which_der=None):
        if kernel_spec is None:
            kernel_spec = {'name': 'rbf', 'params': np.ones((1, dim_in + 1))}
        if point_spec is None:
            point_spec = {'name': 'ut', 'params': None}
        # tell parent to create some dummy gp model, with dummy kernel and dummy points
        super(GaussianProcessDerTransform, self).__init__(dim_in, dim_out, 'gp', kernel_spec, point_spec, estimate_par)
        # NOTE: better solution would be to pass the Model instance directly into the BQTransform
        # overwrite model created in the BQTransform.__init__()
        self.model = GaussianProcessDerModel(dim_in, kernel_spec, point_spec, estimate_par, which_der)
        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self.weights(kernel_spec['params'])

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

    def __init__(self, dim, kernel_spec=None, point_spec=None, estimate_par=False, which_der=None):
        super(GaussianProcessDerModel, self).__init__(dim, kernel_spec, point_spec, estimate_par)
        # overwrite default kernel of the GaussianProcessModel
        self.kernel = RBFGaussDer(dim, kernel_spec['params'])
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
    """RBF kernel "with derivatives". Kernel expectations are w.r.t. Gaussian density."""

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

        # NOTE: benchmark vs. np.kron(), replace with np.kron() if possible, but which_der complicates the matter
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

