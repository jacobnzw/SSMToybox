import numpy as np
from GPy.kern import RBF
from numpy import newaxis as na
from numpy.linalg import det, inv, cholesky
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from transform import MomentTransform


class GPQuad(MomentTransform):
    # GPQ can work with any sigmas so it's probably better to pass in the unit sigmas
    # as an argument instead of creating them in init
    # BQ does not prescribe any sigma-point schemes (apart from minimium variance point sets)
    def __init__(self, unit_sp, hypers=None):
        self.unit_sp = unit_sp  # (d, n)
        # get number of sigmas (n) and dimension of sigmas (d)
        self.d, self.n = self.unit_sp.shape
        # set kernel hyper-parameters (manually or some principled method)
        self.hypers = self._min_var_hypers() if hypers is None else hypers
        # BQ weights given the unit sigma-points and the kernel hyper-parameters
        self.wm, self.Wc, self.Wcc = self.weights_rbf()
        # GPy RBF kernel with given hypers
        self.kern = RBF(self.d, variance=hypers['sig_var'], lengthscale=hypers['lengthscale'], ARD=True)

    def apply(self, f, mean, cov, pars):
        mean = mean[:, na]
        # form sigma-points from unit sigma-points
        x = mean + cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(f, 0, x, pars)
        # output mean
        mean_f = fx.dot(self.wm)
        # output covariance
        cov_f = fx.dot(self.Wc).dot(fx.T) - np.outer(mean_f, mean_f.T)
        cov_f += self.model_var
        # input-output covariance
        cov_fx = self.D.dot(x - mean).dot(self.Wcc).dot(fx.T)
        return mean_f, cov_f, cov_fx.T

    def weights_rbf(self):
        # BQ weights for RBF kernel with given hypers, computations adopted from the GP-ADF code [Deisenroth] with
        # the following assumptions:
        #   (A1) the uncertain input is zero-mean with unit covariance
        #   (A2) one set of hyper-parameters is used for all output dimensions (one GP models all outputs)
        d, n = self.unit_sp.shape
        # GP kernel hyper-parameters
        alpha, el, jitter = self.hypers['sig_var'], self.hypers['lengthscale'], self.hypers['noise_var']
        assert len(el) == d
        # pre-allocation for convenience
        eye_d, eye_n = np.eye(d), np.eye(n)
        iLam1 = np.atleast_2d(np.diag(el ** -1))  # sqrt(Lambda^-1)
        iLam2 = np.atleast_2d(np.diag(el ** -2))

        inp = self.unit_sp.T.dot(iLam1)  # sigmas / el[:, na] (x - m)^T*sqrt(Lambda^-1) # (numSP, xdim)
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
        wm_f = q.dot(iK)
        iKQ = iK.dot(t * L)
        # covariance weights
        wc_f = iKQ.dot(iK)
        # cross-covariance "weights"
        wc_fx = np.diag(q).dot(iK)
        # used for self.D.dot(x - mean).dot(wc_fx).dot(fx)
        self.D = inv(eye_d + np.diag(el ** 2))  # S(S+Lam)^-1; for S=I, (I+Lam)^-1
        # model variance; to be added to the covariance
        # this diagonal form assumes independent GP outputs (cov(f^a, f^b) = 0 for all a, b: a neq b)
        self.model_var = np.diag((alpha ** 2 - np.trace(iKQ)) * np.ones((d, 1)))
        return wm_f, wc_f, wc_fx

    def plot_gp_model(self, f, unit_sp, args):
        fx = np.apply_along_axis(f, 0, unit_sp, args)
        # TODO: plotting w/o GPy's routines
        # TODO: which output dimension to plot, what about n-D inputs?
        from GPy.models import GPRegression
        m = GPRegression(unit_sp.T, fx.T, kernel=self.kern, noise_var=self.hypers['noise_var'])
        m.plot()

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


class GPQuadAlt(GPQuad):
    def apply(self, f, mean, cov, pars):
        # this variant of the GPQuad recomputes weights based on moments (computationally costly)
        pass


class TPQuad(MomentTransform):
    def __init__(self, unit_sp, hypers=None, nu=3.0):
        self.unit_sp = unit_sp  # (d, n)
        # get number of sigmas (n) and dimension of sigmas (d)
        self.d, self.n = self.unit_sp.shape
        # set t-distribution's degrees of freedom parameter nu
        self.nu = nu
        # set kernel hyper-parameters (manually or some principled method)
        self.hypers = self._min_var_hypers() if hypers is None else hypers
        # BQ weights given the unit sigma-points and the kernel hyper-parameters
        self.wm, self.Wc, self.Wcc = self.weights_rbf()
        # GPy RBF kernel with given hypers
        self.kern = RBF(self.d, variance=hypers['sig_var'], lengthscale=hypers['lengthscale'], ARD=True)

    def apply(self, f, mean, cov, pars):
        mean = mean[:, na]
        # form sigma-points from unit sigma-points
        x = mean + cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(f, 0, x, pars)
        # output mean
        mean_f = fx.dot(self.wm)
        # output covariance
        cov_f = fx.dot(self.Wc).dot(fx.T) - np.outer(mean_f, mean_f.T)
        scale = (self.nu - 2 + fx.dot(self.iK).dot(fx.T)) / (self.nu - 2 + self.n)
        cov_f += scale * self.model_var
        # input-output covariance
        dx = x - mean
        cov_fx = self.D.dot(dx).dot(self.Wcc).dot(fx.T)
        return mean_f, cov_f, cov_fx.T

    def weights_rbf(self):
        # TODO: hypers as input argument (in case they're estimated online)
        # BQ weights for RBF kernel with given hypers, computations adopted from the GP-ADF code [Deisenroth] with
        # the following assumptions:
        #   (A1) the uncertain input is zero-mean with unit covariance
        #   (A2) one set of hyper-parameters is used for all output dimensions (one GP models all outputs)
        d, n = self.unit_sp.shape
        # GP kernel hyper-parameters
        alpha, el, jitter = self.hypers['sig_var'], self.hypers['lengthscale'], self.hypers['noise_var']
        assert len(el) == d
        # pre-allocation for convenience
        eye_d, eye_n = np.eye(d), np.eye(n)
        iLam1 = np.atleast_2d(np.diag(el ** -1))  # sqrt(Lambda^-1)
        iLam2 = np.atleast_2d(np.diag(el ** -2))

        inp = self.unit_sp.T.dot(iLam1)  # sigmas / el[:, na] (x - m)^T*sqrt(Lambda^-1) # (numSP, xdim)
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
        wm_f = q.dot(iK)
        iKQ = iK.dot(t * L)
        # covariance weights
        wc_f = iKQ.dot(iK)
        # cross-covariance "weights"
        wc_fx = np.diag(q).dot(iK)
        self.iK = iK
        # used for self.D.dot(x - mean).dot(wc_fx).dot(fx)
        self.D = inv(eye_d + np.diag(el ** 2))  # S(S+Lam)^-1; for S=I, (I+Lam)^-1
        # model variance; to be added to the covariance
        # this diagonal form assumes independent GP outputs (cov(f^a, f^b) = 0 for all a, b: a neq b)
        self.model_var = np.diag((alpha ** 2 - np.trace(iKQ)) * np.ones((d, 1)))
        return wm_f, wc_f, wc_fx


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

# TODO: add GPQ+D(derivative observations),
# TODO: add GPQ+TD (total derivative observations)
# TODO: add GPQ+DIV (divergence observations)
