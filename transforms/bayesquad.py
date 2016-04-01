import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from GPy.kern import RBF, Linear, Bias
from numpy import newaxis as na
from numpy.linalg import det, inv
from numpy.polynomial.hermite_e import hermeval
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from transform import BayesianQuadratureTransform


def maha(x, y, V=None):
    """
    Pair-wise Mahalanobis distance of rows of x and y with given weight matrix V.
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


class GPQuad(BayesianQuadratureTransform):
    def __init__(self, dim, unit_sp=None, hypers=None):
        super(GPQuad, self).__init__(dim, unit_sp, hypers)
        # GPy RBF kernel with given hypers
        self.kern = RBF(self.d, variance=self.hypers['sig_var'], lengthscale=self.hypers['lengthscale'], ARD=True)

    def weights_rbf(self, unit_sp, hypers):
        # BQ weights for RBF kernel with given hypers, computations adopted from the GP-ADF code [Deisenroth] with
        # the following assumptions:
        #   (A1) the uncertain input is zero-mean with unit covariance
        #   (A2) one set of hyper-parameters is used for all output dimensions (one GP models all outputs)
        d, n = unit_sp.shape
        # GP kernel hyper-parameters
        alpha, el, jitter = hypers['sig_var'], hypers['lengthscale'], hypers['noise_var']
        assert len(el) == d
        # pre-allocation for convenience
        eye_d, eye_n = np.eye(d), np.eye(n)
        iLam1 = np.atleast_2d(np.diag(el ** -1))  # sqrt(Lambda^-1)
        iLam2 = np.atleast_2d(np.diag(el ** -2))

        inp = unit_sp.T.dot(iLam1)  # sigmas / el[:, na] (x - m)^T*sqrt(Lambda^-1) # (numSP, xdim)
        K = np.exp(2 * np.log(alpha) - 0.5 * maha(inp, inp))
        iK = cho_solve(cho_factor(K + jitter * eye_n), eye_n)
        B = iLam2 + eye_d  # (D, D)
        c = alpha ** 2 / np.sqrt(det(B))
        t = inp.dot(inv(B))  # inn*(P + Lambda)^-1
        l = np.exp(-0.5 * np.sum(inp * t, 1))  # (N, 1)
        zet = 2 * np.log(alpha) - 0.5 * np.sum(inp * inp, 1)
        inp = inp.dot(iLam1)
        R = 2 * iLam2 + eye_d
        t = 1 / np.sqrt(det(R))
        L = np.exp((zet[:, na] + zet[:, na].T) + maha(inp, -inp, V=0.5 * inv(R)))
        q = c * l  # evaluations of the kernel mean map (from the viewpoint of RHKS methods)
        # mean weights
        wm = q.dot(iK)
        iKQ = iK.dot(t * L)
        # covariance weights
        Wc = iKQ.dot(iK)
        # cross-covariance "weights"
        mu_q = inv(np.diag(el ** 2) + eye_d).dot(unit_sp)  # (d, n)
        Wcc = (q[na, :] * mu_q).dot(iK)  # (d, n)
        # model variance; to be added to the covariance
        # this diagonal form assumes independent GP outputs (cov(f^a, f^b) = 0 for all a, b: a neq b)
        self.model_var = np.diag((alpha ** 2 - np.trace(iKQ)) * np.ones((d, 1)))
        return wm, Wc, Wcc

    def plot_gp_model(self, f, unit_sp, args, test_range=(-5, 5, 50), plot_dims=(0, 0)):
        # plot out_dim vs. in_dim
        in_dim, out_dim = plot_dims
        # test input must have the same dimension as specified in kernel
        test = np.linspace(*test_range)
        test_pts = np.zeros((self.d, len(test)))
        test_pts[in_dim, :] = test
        # function value observations at training points (unit sigma-points)
        y = np.apply_along_axis(f, 0, unit_sp, args)
        fx = np.apply_along_axis(f, 0, test_pts, args)  # function values at test points
        K = self.kern.K(unit_sp.T)  # covariances between sigma-points
        k = self.kern.K(test_pts.T, unit_sp.T)  # covariance between test inputs and sigma-points
        kxx = self.kern.Kdiag(test_pts.T)  # prior predictive variance
        k_iK = cho_solve(cho_factor(K), k.T).T
        gp_mean = k_iK.dot(y[out_dim, :])  # GP mean
        gp_var = np.diag(np.diag(kxx) - k_iK.dot(k.T))  # GP predictive variance
        # plot the GP mean, predictive variance and the true function
        plt.figure()
        plt.plot(test, fx[out_dim, :], color='r', ls='--', lw=2, label='true')
        plt.plot(test, gp_mean, color='b', ls='-', lw=2, label='GP mean')
        plt.fill_between(test, gp_mean + 2 * np.sqrt(gp_var), gp_mean - 2 * np.sqrt(gp_var),
                         color='b', alpha=0.25, label='GP variance')
        plt.plot(unit_sp[in_dim, :], y[out_dim, :],
                 color='k', ls='', marker='o', ms=8, label='data')
        plt.legend()
        plt.show()

    def default_sigma_points(self, dim):
        # create unscented points
        c = np.sqrt(dim)
        return np.hstack((np.zeros((dim, 1)), c * np.eye(dim), -c * np.eye(dim)))

    def default_hypers(self, dim):
        # define default hypers
        return {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(dim, ), 'noise_var': 1e-8}

    def _weights(self, sigma_points, hypers):
        return self.weights_rbf(sigma_points, hypers)

    def _fcn_eval(self, fcn, x, fcn_pars):
        return np.apply_along_axis(fcn, 0, x, fcn_pars)

    def _int_var_rbf(self, X, hyp, jitter=1e-8):
        """
        Posterior integral variance of the Gaussian Process quadrature.
        X - vector (1, 2*xdim**2+xdim)
        hyp - kernel hyperparameters [s2, el_1, ... el_d]
        """
        # reshape X to SP matrix
        X = np.reshape(X, (self.n, self.d))
        # set kernel hyper-parameters
        s2, el = hyp[0], hyp[1:]
        self.kern.param_array[0] = s2  # variance
        self.kern.param_array[1:] = el  # lengthscale
        K = self.kern.K(X)
        L = np.diag(el ** 2)
        # posterior variance of the integral
        ks = s2 * np.sqrt(det(L + np.eye(self.d))) * multivariate_normal(mean=np.zeros(self.d), cov=L).pdf(X)
        postvar = -ks.dot(solve(K + jitter * np.eye(self.n), ks.T))
        return postvar

    def _int_var_rbf_hyp(self, hyp, X, jitter=1e-8):
        """
        Posterior integral variance as a function of hyper-parameters
        :param hyp: RBF kernel hyper-parameters [s2, el_1, ..., el_d]
        :param X: sigma-points
        :param jitter: numerical jitter (for stabilizing computations)
        :return: posterior integral variance
        """
        # reshape X to SP matrix
        X = np.reshape(X, (self.n, self.d))
        # set kernel hyper-parameters
        s2, el = 1, hyp  # sig_var hyper always set to 1
        self.kern.param_array[0] = s2  # variance
        self.kern.param_array[1:] = el  # lengthscale
        K = self.kern.K(X)
        L = np.diag(el ** 2)
        # posterior variance of the integral
        ks = s2 * np.sqrt(det(L + np.eye(self.d))) * multivariate_normal(mean=np.zeros(self.d), cov=L).pdf(X)
        postvar = s2 * np.sqrt(det(2 * inv(L) + np.eye(self.d))) ** -1 - ks.dot(
            solve(K + jitter * np.eye(self.n), ks.T))
        return postvar

    def _min_var_sigmas(self):
        # solver options
        op = {'disp': True}
        # bounds based on input unit Gaussian (-2*std, +2std)
        bnds = tuple((-2, 2) for i in range(self.n * self.d))
        hyp = np.hstack((self.hypers['sig_var'], self.hypers['lengthscale']))
        # unconstrained
        #        res = minimize(self._gpq_postvar, self.X0, method='Nelder-Mead', options=op)
        #        res = minimize(self._gpq_postvar, self.X0, method='SLSQP', bounds=bnds, options=op)
        res = minimize(self._int_var_rbf, self.unit_sp, args=hyp, method='L-BFGS-B', bounds=bnds, options=op)
        return res.x

    def _min_var_hypers(self):
        """
        Finds kernel hyper-parameters minimizing the posterior integral variance.
        :return: optimized kernel hyper-parameters
        """
        # solver options
        op = {'disp': True}
        hyp = self.hypers['lengthscale']  # np.hstack((self.hypers['sig_var'], self.hypers['lengthscale']))
        # bounds based on input unit Gaussian (-2*std, +2std)
        bnds = tuple((1e-3, 1000) for i in range(len(hyp)))
        # unconstrained
        #        res = minimize(self._gpq_postvar, self.X0, method='Nelder-Mead', options=op)
        #        res = minimize(self._gpq_postvar, self.X0, method='SLSQP', bounds=bnds, options=op)
        res = minimize(self._int_var_rbf_hyp, hyp, args=self.unit_sp, method='L-BFGS-B', bounds=bnds, options=op)
        return res.x

    def _min_logmarglik_hypers(self):
        # finds hypers by maximizing the marginal likelihood (empirical bayes)
        # the multiple output dimensions should be reflected in the log marglik
        pass

    def _min_intvar_logmarglik_hypers(self):
        # finds hypers by minimizing the sum of log-marginal likelihood and the integral variance objectives
        pass


class GPQuadDerAffine(BayesianQuadratureTransform):
    """
    Gaussian Process Quadrature with affine kernel which uses derivative observations (in addition to function values).
    """

    def __init__(self, dim, unit_sp=None, hypers=None, which_der=None):
        super(GPQuadDerAffine, self).__init__(dim, unit_sp, hypers)
        # get number of sigmas (n) and dimension of sigmas (d)
        self.d, self.n = self.unit_sp.shape
        # assume derivatives evaluated at all sigmas if unspecified
        self.which_der = which_der if which_der is not None else np.arange(self.n)
        # GPy Linear + Bias kernel with given hypers
        self.kern = Linear(dim, variances=hypers['variance']) + Bias(dim, variance=hypers['bias'])

    def weights_affine(self, unit_sp, hypers):
        d, n = unit_sp.shape
        # GP kernel hyper-parameters
        alpha, el, jitter = hypers['bias'], hypers['variance'], hypers['noise_var']
        assert len(el) == d
        # pre-allocation for convenience
        eye_d, eye_n, eye_y = np.eye(d), np.eye(n), np.eye(n + d * n)
        Lam = np.diag(el ** 2)

        K = self.kern_affine_der(unit_sp, hypers)  # evaluate kernel matrix BOTTLENECK
        iK = cho_solve(cho_factor(K + jitter * eye_y), eye_y)  # invert kernel matrix BOTTLENECK
        q_tilde = np.hstack((alpha ** 2 * np.ones(n), np.zeros(n * d)))
        # weights for mean
        wm = q_tilde.dot(iK)

        #  quantities for cross-covariance "weights"
        R_tilde = np.hstack((Lam.dot(unit_sp), np.tile(Lam, (1, n))))  # (D, N+N*D)
        # input-output covariance (cross-covariance) "weights"
        Wcc = R_tilde.dot(iK)  # (D, N+N*D)  # FIXME: weights still seem fishy, not symmetric etc.
        # expectations of products of kernels
        E_ff_ff = alpha ** 2 + unit_sp.T.dot(Lam).dot(Lam).dot(unit_sp)
        E_ff_fd = np.tile(unit_sp.T.dot(Lam).dot(Lam), (1, n))
        E_df_fd = np.tile(Lam.dot(Lam), (n, n))
        Q_tilde = np.vstack((np.hstack((E_ff_ff, E_ff_fd)), np.hstack((E_ff_fd.T, E_df_fd))))

        # weights for covariance
        iKQ = iK.dot(Q_tilde)
        Wc = iKQ.dot(iK)

        # model variance
        self.model_var = np.diag((alpha ** 2 + np.trace(Lam) - np.trace(iKQ)) * np.ones((d, 1)))
        assert self.model_var >= 0
        return wm, Wc, Wcc

    @staticmethod
    def kern_affine_der(X, hypers):
        d, n = X.shape
        # extract hypers
        alpha, el, jitter = hypers['bias'], hypers['variance'], hypers['noise_var']
        assert len(el) == d
        Lam = np.diag(el ** 2)
        Kff = alpha ** 2 + X.T.dot(Lam).dot(X)
        Kfd = np.tile(X.T.dot(Lam), (1, n))
        Kdd = np.tile(Lam, (n, n))
        return np.vstack((np.hstack((Kff, Kfd)), np.hstack((Kfd.T, Kdd))))

    def plot_gp_model(self, f, unit_sp, args, test_range=(-5, 5, 50), plot_dims=(0, 0)):
        # TODO: modify kernel evaluations so that the GP fit accounts for derivative information
        # plot out_dim vs. in_dim
        in_dim, out_dim = plot_dims
        # test input must have the same dimension as specified in kernel
        test = np.linspace(*test_range)
        test_pts = np.zeros((self.d, len(test)))
        test_pts[in_dim, :] = test
        # function value observations at training points (unit sigma-points)
        y = np.apply_along_axis(f, 0, unit_sp, args)
        fx = np.apply_along_axis(f, 0, test_pts, args)  # function values at test points
        K = self.kern.K(unit_sp.T)  # covariances between sigma-points
        k = self.kern.K(test_pts.T, unit_sp.T)  # covariance between test inputs and sigma-points
        kxx = self.kern.Kdiag(test_pts.T)  # prior predictive variance
        k_iK = cho_solve(cho_factor(K), k.T).T
        gp_mean = k_iK.dot(y[out_dim, :])  # GP mean
        gp_var = np.diag(np.diag(kxx) - k_iK.dot(k.T))  # GP predictive variance
        # plot the GP mean, predictive variance and the true function
        plt.figure()
        plt.plot(test, fx[out_dim, :], color='r', ls='--', lw=2, label='true')
        plt.plot(test, gp_mean, color='b', ls='-', lw=2, label='GP mean')
        plt.fill_between(test, gp_mean + 2 * np.sqrt(gp_var), gp_mean - 2 * np.sqrt(gp_var),
                         color='b', alpha=0.25, label='GP variance')
        plt.plot(unit_sp[in_dim, :], y[out_dim, :],
                 color='k', ls='', marker='o', ms=8, label='data')
        plt.legend()
        plt.show()

    def default_sigma_points(self, dim):
        # one sigma-point
        return np.zeros((dim, 1))

    def default_hypers(self, dim):
        # define default hypers
        return {'bias': 1.0, 'variance': 1.0 * np.ones(dim, ), 'noise_var': 1e-8}

    def _weights(self, sigma_points, hypers):
        return self.weights_affine(sigma_points, hypers)

    def _fcn_eval(self, fcn, x, fcn_pars):
        # should return as many columns as output dims, one column includes function and derivative evaluations
        # for every sigma-point, thus it is (n + n*d,); n = # sigma-points, d = sigma-point dimensionality
        # fx should be (n + n*d, e); e = output dimensionality
        # evaluate function at sigmas (e, n)
        fx = np.apply_along_axis(fcn, 0, x, fcn_pars)
        # Jacobians evaluated only at sigmas specified by which_der array
        dfx = np.zeros((fx.shape[0] * self.d, self.n))
        dfx[:, self.which_der] = np.apply_along_axis(fcn, 0, x[:, self.which_der], fcn_pars, dx=True)
        # stack function values and derivative values into one column
        return np.vstack((fx.T, dfx.T.reshape(self.d * self.n, -1))).T


class GPQuadDerRBF(BayesianQuadratureTransform):
    """
    Gaussian Process Quadrature with RBF kernel which uses derivative observations (in addition to function values).
    """

    def __init__(self, dim, unit_sp=None, hypers=None, which_der=None):
        super(GPQuadDerRBF, self).__init__(dim, unit_sp, hypers)
        # get number of sigmas (n) and dimension of sigmas (d)
        self.d, self.n = self.unit_sp.shape
        # assume derivatives evaluated at all sigmas if unspecified
        self.which_der = which_der if which_der is not None else np.arange(self.n)
        # GPy RBF kernel with given hypers
        self.kern = RBF(self.d, variance=self.hypers['sig_var'], lengthscale=self.hypers['lengthscale'], ARD=True)

    def weights_rbf(self, unit_sp, hypers):
        d, n = unit_sp.shape
        # GP kernel hyper-parameters
        alpha, el, jitter = hypers['sig_var'], hypers['lengthscale'], hypers['noise_var']
        assert len(el) == d
        # pre-allocation for convenience
        eye_d, eye_n, eye_y = np.eye(d), np.eye(n), np.eye(n + d * n)

        K = self.kern_rbf_der(unit_sp, hypers)  # evaluate kernel matrix BOTTLENECK
        iK = cho_solve(cho_factor(K + jitter * eye_y), eye_y)  # invert kernel matrix BOTTLENECK
        Lam = np.diag(el ** 2)
        iLam = np.diag(el ** -1)  # sqrt(Lambda^-1)
        iiLam = np.diag(el ** -2)  # Lambda^-1
        inn = iLam.dot(unit_sp)  # (x-m)^T*iLam  # (N, D)
        B = iiLam + eye_d  # P*Lambda^-1+I, (P+Lam)^-1 = Lam^-1*(P*Lam^-1+I)^-1 # (D, D)
        cho_B = cho_factor(B)
        t = cho_solve(cho_B, inn)  # dot(inn, inv(B)) # (x-m)^T*iLam*(P+Lambda)^-1  # (D, N)
        l = np.exp(-0.5 * np.sum(inn * t, 0))  # (N, 1)
        q = (alpha ** 2 / np.sqrt(det(B))) * l  # (N, 1)
        Sig_q = cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(unit_sp)  # (D,N) Sig_q*x
        mu_q = iiLam.dot(eta)  # (D,N)
        r = q[na, :] * iiLam.dot(mu_q - unit_sp)  # -t.dot(iLam) * q  # (D, N)
        q_tilde = np.hstack((q.T, r.T.ravel()))  # (1, N+N*D)

        # weights for mean
        wm = q_tilde.dot(iK)

        #  quantities for cross-covariance "weights"
        iLamSig = iiLam.dot(Sig_q)  # (D,D)
        r_tilde = (q[na, na, :] * iLamSig[..., na] + mu_q[na, ...] * r[:, na, :]).T.reshape(n * d, d).T  # (D, N*D)
        R_tilde = np.hstack((q[na, :] * mu_q, r_tilde))  # (D, N+N*D)

        # input-output covariance (cross-covariance) "weights"
        Wcc = R_tilde.dot(iK)  # (D, N+N*D)

        # quantities for covariance weights
        zet = 2 * np.log(alpha) - 0.5 * np.sum(inn * inn, 0)  # (D,N) 2log(alpha) - 0.5*(x-m)^T*Lambda^-1*(x-m)
        inn = iiLam.dot(unit_sp)  # inp / el[:, na]**2
        R = 2 * iiLam + eye_d  # 2P*Lambda^-1 + I
        # (N,N)
        Q = (1.0 / np.sqrt(det(R))) * np.exp((zet[:, na] + zet[:, na].T) + maha(inn.T, -inn.T, V=0.5 * solve(R, eye_d)))
        cho_LamSig = cho_factor(Lam + Sig_q)
        Sig_Q = cho_solve(cho_LamSig, Sig_q).dot(iiLam)  # (D,D) Lambda^-1 (Lambda*(Lambda+Sig_q)^-1*Sig_q) Lambda^-1
        eta_tilde = iiLam.dot(cho_solve(cho_LamSig, eta))  # Lambda^-1(Lambda+Sig_q)^-1*eta
        ETA = eta_tilde[..., na] + eta_tilde[:, na, :]  # (D,N,N) pairwise sum of pre-multiplied eta's (D,N,N)
        # mu_Q = ETA + in_mean[:, na]  # (D,N,N)
        xnmu = inn[..., na] - ETA  # (D,N,N) x_n - mu^Q_nm
        # xmmu = sigmas[:, na, :] - mu_Q  # x_m - mu^Q_nm
        E_dff = (-Q[na, ...] * xnmu).swapaxes(0, 1).reshape(d * n, n)
        # (D,D,N,N) (x_n - mu^Q_nm)(x_m - mu^Q_nm)^T + Sig_Q
        T = xnmu[:, na, ...] * xnmu.swapaxes(1, 2)[na, ...] + Sig_Q[..., na, na]
        E_dffd = (Q[na, na, ...] * T).swapaxes(0, 3).reshape(d * n, -1)  # (N*D, N*D)
        Q_tilde = np.vstack((np.hstack((Q, E_dff.T)), np.hstack((E_dff, E_dffd))))  # (N+N*D, N+N*D)

        # weights for covariance
        iKQ = iK.dot(Q_tilde)
        Wc = iKQ.dot(iK)
        # model variance
        self.model_var = np.diag((alpha ** 2 - np.trace(iKQ)) * np.ones((d, 1)))
        return wm, Wc, Wcc

    @staticmethod
    def kern_rbf_der(X, hypers):
        # TODO: rewrite in Cython, try Numba @jit decorator and get rid of double loops
        D, N = X.shape
        # extract hypers
        alpha, el, jitter = hypers['sig_var'], hypers['lengthscale'], hypers['noise_var']
        iLam = np.diag(el ** -1 * np.ones(D))
        iiLam = np.diag(el ** -2 * np.ones(D))

        X = iLam.dot(X)  # sqrt(Lambda^-1) * X
        Kff = np.exp(2 * np.log(alpha) - 0.5 * maha(X.T, X.T))  # cov(f(xi), f(xj))
        X = iLam.dot(X)  # Lambda^1 * X
        XmX = X[..., na] - X[:, na, :]
        Kfd = np.zeros((N, D * N))  # cov(f(xi), df(xj))
        Kdd = np.zeros((D * N, D * N))  # cov(df(xi), df(xj))
        for i in range(N):
            for j in range(N):
                istart, iend = i * D, i * D + D
                jstart, jend = j * D, j * D + D
                Kfd[i, jstart:jend] = Kff[i, j] * XmX[:, i, j]
                Kdd[istart:iend, jstart:jend] = Kff[i, j] * (iiLam - np.outer(XmX[:, i, j], XmX[:, i, j]))
        # Kdd = (Kdd + Kdd.T)
        return np.vstack((np.hstack((Kff, Kfd)), np.hstack((Kfd.T, Kdd))))

    def plot_gp_model(self, f, unit_sp, args, test_range=(-5, 5, 50), plot_dims=(0, 0)):
        # TODO: modify kernel evaluations so that the GP fit accounts for derivative information
        # plot out_dim vs. in_dim
        in_dim, out_dim = plot_dims
        # test input must have the same dimension as specified in kernel
        test = np.linspace(*test_range)
        test_pts = np.zeros((self.d, len(test)))
        test_pts[in_dim, :] = test
        # function value observations at training points (unit sigma-points)
        y = np.apply_along_axis(f, 0, unit_sp, args)
        fx = np.apply_along_axis(f, 0, test_pts, args)  # function values at test points
        K = self.kern.K(unit_sp.T)  # covariances between sigma-points
        k = self.kern.K(test_pts.T, unit_sp.T)  # covariance between test inputs and sigma-points
        kxx = self.kern.Kdiag(test_pts.T)  # prior predictive variance
        k_iK = cho_solve(cho_factor(K), k.T).T
        gp_mean = k_iK.dot(y[out_dim, :])  # GP mean
        gp_var = np.diag(np.diag(kxx) - k_iK.dot(k.T))  # GP predictive variance
        # plot the GP mean, predictive variance and the true function
        plt.figure()
        plt.plot(test, fx[out_dim, :], color='r', ls='--', lw=2, label='true')
        plt.plot(test, gp_mean, color='b', ls='-', lw=2, label='GP mean')
        plt.fill_between(test, gp_mean + 2 * np.sqrt(gp_var), gp_mean - 2 * np.sqrt(gp_var),
                         color='b', alpha=0.25, label='GP variance')
        plt.plot(unit_sp[in_dim, :], y[out_dim, :],
                 color='k', ls='', marker='o', ms=8, label='data')
        plt.legend()
        plt.show()

    def default_sigma_points(self, dim):
        # create unscented points
        c = np.sqrt(dim)
        return np.hstack((np.zeros((dim, 1)), c * np.eye(dim), -c * np.eye(dim)))

    def default_hypers(self, dim):
        # define default hypers
        return {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(dim, ), 'noise_var': 1e-8}

    def _weights(self, sigma_points, hypers):
        return self.weights_rbf(sigma_points, hypers)

    def _fcn_eval(self, fcn, x, fcn_pars):
        # should return as many columns as output dims, one column includes function and derivative evaluations
        # for every sigma-point, thus it is (n + n*d,); n = # sigma-points, d = sigma-point dimensionality
        # fx should be (n + n*d, e); e = output dimensionality
        # evaluate function at sigmas (e, n)
        fx = np.apply_along_axis(fcn, 0, x, fcn_pars)
        # Jacobians evaluated only at sigmas specified by which_der array
        dfx = np.zeros((fx.shape[0] * self.d, self.n))
        dfx[:, self.which_der] = np.apply_along_axis(fcn, 0, x[:, self.which_der], fcn_pars, dx=True)
        # stack function values and derivative values into one column
        return np.vstack((fx.T, dfx.T.reshape(self.d * self.n, -1))).T


class GPQuadDerHermite(BayesianQuadratureTransform):
    def __init__(self, dim, unit_sp=None, hypers=None, which_der=None):
        super(GPQuadDerHermite, self).__init__(dim, unit_sp, hypers)
        # get number of sigmas (n) and dimension of sigmas (d)
        self.d, self.n = self.unit_sp.shape
        # assume derivatives evaluated at all sigmas if unspecified
        self.which_der = which_der if which_der is not None else np.arange(self.n)
        # GPy RBF kernel with given hypers
        # self.kern = RBF(self.d, variance=self.hypers['sig_var'], lengthscale=self.hypers['lengthscale'], ARD=True)

    def default_sigma_points(self, dim):
        # create unscented points
        c = np.sqrt(dim)
        return np.hstack((np.zeros((dim, 1)), c * np.eye(dim), -c * np.eye(dim)))

    def default_hypers(self, dim):
        # define default hypers
        return {'lambdas': np.ones(4), 'noise_var': 1e-8}

    def weights_hermite(self, unit_sp, hypers):
        pass

    def kern_hermite_der(self, X, hypers):
        lamb = hypers['lambda']
        kff = self._kernel_ut(X, X, lamb)
        kfd, kdd = self._kernel_ut_dx(X, X, lamb)
        return np.vstack((np.hstack((kff, kfd)), np.hstack((kfd.T, kdd))))

    @staticmethod
    def multihermite(x, ind):
        """Evaluate multivariate Hermite polynomial.

        Evaluate multi-index Hermite polynomial with indices ind at columns in matrix x.

        Parameters
        ----------
        x : 2-D array_like

        ind : 1-D array_like
            multi-index specifying degree of univariate Hermite polynomials for each input dimension
        Returns
        -------
        1-D array_like of shape (n,)

        """
        x, ind = np.atleast_2d(x), np.atleast_1d(ind)
        d, n = x.shape
        ind[ind < 0] = 0  # convert negative degrees to 0-th degree
        p = np.max(ind)
        order = np.arange(p + 1)
        y = np.ones(n)
        for i in range(d):
            pmask = (order == ind[i]).astype(int)
            y = y * hermeval(x[i, :], pmask)
        return y

    @staticmethod
    @jit
    def ind_sum(n, p):
        """All possible sets of size n containing positive integers summing to p.

        Returns sets organized into columns of (n, n**p) matrix.
        Original MATLAB code by Simo Sarkka
        Python code by me.
        """
        if p == 0:
            iset = np.zeros((n, 1))
        elif p == 1:
            iset = np.eye(n)
        else:
            iset1 = GPQuadDerHermite.ind_sum(n, p - 1)  # (n, n**(p-1))
            iset2 = np.eye(n)  # (n, n)
            iset = np.zeros((n, n ** p))
            k = 0
            for i in range(n ** (p - 1)):
                for j in range(n):
                    iset[:, k] = iset1[:, i] + iset2[:, j]
                    k += 1
        return iset

    @staticmethod
    def multifactorial(multi_index):
        return np.apply_along_axis(np.math.factorial, 0, np.atleast_2d(multi_index)).prod()

    def _weights(self, sigma_points, hypers):
        return None, None, None

    # @jit
    def _kernel_ut(self, x, xs, lamb=np.ones(4)):
        """Unscented transform covariance function (kernel).

        Unscented transform covariance function is given by
        ..math::
        $ k(\mathbf{x}, \mathbf{x}) = \sum_{p=0}^{3}\sum_{q=0}^{3}\sum_{|I|=p}\sum_{|J|=q}
        (\mathcal{I}!\mathcal{J}!)^{-1} \lambda_{\mathcal{I}, \mathcal{J}}
        H_{\mathcal{I}}(\mathbf{x})H_{\mathcal{J}}(\mathbf{x}^\prime) $,
        where ..math:: $ \lambda_{\mathcal{I}, \mathcal{J}} $ are the function hyper-parameters. This form is impractical,
        because there are
        ..math:: $ \sum_{p=0}^{3}\sum_{q=0}^{3} d^{p + q} $
        hyperparameters, .

        For practical reasons, this function implements an approximation of the above UT covariance function, given as
        ..math::
        $ k(\mathbf{x}, \mathbf{x}) \approxeq \sum_{p=0}^{3}\sum_{|I|=p}
        (\mathcal{I}!)^{-2} \lambda_p H_{\mathcal{I}}(\mathbf{x})H_{\mathcal{I}}(\mathbf{x}^\prime) $,
        which reduces the number of hyper-parameters to 4.

        Parameters
        ----------
        x: 2-D numpy.ndarray
            Training inputs
        xs: 2-D numpy.ndarray
            Test inputs
        lamb: 1-D numpy.ndarray
            Hyper-parameters

        Returns
        -------

        """
        d, n = x.shape
        e, m = xs.shape
        assert d == e
        assert lamb.ndim == 1
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):  # TODO: no need for these because multihermite can take care of this
                for p in range(4):
                    iset = self.ind_sum(d, p)
                    h = 0
                    for k in range(d ** p):
                        c = self.multifactorial(iset[:, k]) ** -2
                        h += lamb[p] * c * \
                             self.multihermite(x[:, i, na], iset[:, k]) * \
                             self.multihermite(xs[:, j, na], iset[:, k])
                    K[i, j] += h
        return K

    # @jit
    def _kernel_ut_dx(self, x, xs, lamb=np.ones(4)):
        d, n = x.shape
        e, m = xs.shape
        assert d == e
        assert lamb.ndim == 1
        Kfd = np.zeros((n, d * m))  # covariance between functoin value and derivative
        Kdd = np.zeros((d * n, d * m))  # covariance between derivatives
        I = np.eye(d)
        dHi = np.zeros((d,))  # space for gradient
        dHj = dHi.copy()
        for i in range(n):
            for j in range(m):
                # evaluate UT kernel
                for p in range(4):  # for all degrees
                    iset = self.ind_sum(d, p)
                    istart, iend = i * d, i * d + d
                    jstart, jend = j * d, j * d + d
                    for k in range(d ** p):  # for all multi-indexes
                        # evaluate gradient of multivariate Hermite polynomial
                        for dim in range(d):  # for all datapoint dimensions
                            multi_ind = iset[:, k] - I[:, dim]
                            dHi[dim] = self.multihermite(x[:, i, na], multi_ind)
                            dHj[dim] = self.multihermite(xs[:, j, na], multi_ind)
                        dHi *= iset[:, k]  # element-wise product with multi-index
                        dHj *= iset[:, k]
                        c = lamb[p] * self.multifactorial(iset[:, k]) ** -2
                        Kfd[i, jstart:jend] += c * self.multihermite(x[:, i, na], iset[:, k]) * dHj
                        Kdd[istart:iend, jstart:jend] += c * np.outer(dHi, dHj)
        return Kfd, Kdd



class TPQuad(BayesianQuadratureTransform):
    def __init__(self, dim, unit_sp=None, hypers=None, nu=3.0):
        super(TPQuad, self).__init__(dim, unit_sp, hypers)
        # set t-distribution's degrees of freedom parameter nu
        self.nu = nu
        # GPy RBF kernel with given hypers
        self.kern = RBF(self.d, variance=self.hypers['sig_var'], lengthscale=self.hypers['lengthscale'], ARD=True)

    def weights_rbf(self, unit_sp, hypers):
        # BQ weights for RBF kernel with given hypers, computations adopted from the GP-ADF code [Deisenroth] with
        # the following assumptions:
        #   (A1) the uncertain input is zero-mean with unit covariance
        #   (A2) one set of hyper-parameters is used for all output dimensions (one GP models all outputs)
        d, n = unit_sp.shape
        # GP kernel hyper-parameters
        alpha, el, jitter = hypers['sig_var'], hypers['lengthscale'], hypers['noise_var']
        assert len(el) == d
        # pre-allocation for convenience
        eye_d, eye_n = np.eye(d), np.eye(n)
        iLam1 = np.atleast_2d(np.diag(el ** -1))  # sqrt(Lambda^-1)
        iLam2 = np.atleast_2d(np.diag(el ** -2))

        inp = unit_sp.T.dot(iLam1)  # sigmas / el[:, na] (x - m)^T*sqrt(Lambda^-1) # (numSP, xdim)
        K = np.exp(2 * np.log(alpha) - 0.5 * maha(inp, inp))
        iK = cho_solve(cho_factor(K + jitter * eye_n), eye_n)
        B = iLam2 + eye_d  # (D, D)
        c = alpha ** 2 / np.sqrt(det(B))
        t = inp.dot(inv(B))  # inn*(P + Lambda)^-1
        l = np.exp(-0.5 * np.sum(inp * t, 1))  # (N, 1)
        zet = 2 * np.log(alpha) - 0.5 * np.sum(inp * inp, 1)
        inp = inp.dot(iLam1)
        R = 2 * iLam2 + eye_d
        t = 1 / np.sqrt(det(R))
        L = np.exp((zet[:, na] + zet[:, na].T) + maha(inp, -inp, V=0.5 * inv(R)))
        q = c * l  # evaluations of the kernel mean map (from the viewpoint of RHKS methods)
        # mean weights
        wm = q.dot(iK)
        iKQ = iK.dot(t * L)
        # covariance weights
        Wc = iKQ.dot(iK)
        # cross-covariance "weights"
        mu_q = inv(np.diag(el ** 2) + eye_d).dot(unit_sp)  # (d, n)
        Wcc = (q[na, :] * mu_q).dot(iK)  # (d, n)
        self.iK = iK
        # model variance; to be added to the covariance
        # this diagonal form assumes independent GP outputs (cov(f^a, f^b) = 0 for all a, b: a neq b)
        self.model_var = np.diag((alpha ** 2 - np.trace(iKQ)) * np.ones((d, 1)))
        return wm, Wc, Wcc

    def default_sigma_points(self, dim):
        # create unscented points
        c = np.sqrt(dim)
        return np.hstack((np.zeros((dim, 1)), c * np.eye(dim), -c * np.eye(dim)))

    def default_hypers(self, dim):
        # define default hypers
        return {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(dim, ), 'noise_var': 1e-8}

    def _weights(self, sigma_points, hypers):
        return self.weights_rbf(sigma_points, hypers)

    def _fcn_eval(self, fcn, x, fcn_pars):
        return np.apply_along_axis(fcn, 0, x, fcn_pars)

    def _covariance(self, weights, fcn_evals, mean_out):
        scale = (self.nu - 2 + fcn_evals.dot(self.iK).dot(fcn_evals.T)) / (self.nu - 2 + self.n)
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + scale * self.model_var

# TODO: add GPQ+TD (total derivative observations)
# TODO: add GPQ+DIV (divergence observations)
