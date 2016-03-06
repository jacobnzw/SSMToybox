import numpy as np
from numpy import newaxis as na
from numpy.linalg import det, inv, cholesky
from scipy.linalg import cho_factor, cho_solve

from transform import MomentTransform


class GPQuad(MomentTransform):
    # GPQ can work with any sigmas so it's probably better to pass in the unit sigmas
    # as an argument instead of creating them in init
    # BQ does not prescribe any sigma-point schemes (apart from minimium variance point sets)
    def __init__(self, unit_sp, hypers=None):
        self.unit_sp = unit_sp
        # set kernel hyper-parameters (manually or some principled method)
        self.hypers = self._min_var_hypers() if hypers is None else hypers
        # BQ weights given the unit sigma-points and the kernel hyper-parameters
        self.wm, self.Wc, self.Wcc = self.weights_rbf()

    def apply(self, f, mean, cov, *args):
        # form sigma-points from unit sigma-points
        x = mean + cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(f, 0, x, *args)
        # output mean
        mean_f = fx.dot(self.wm)
        # output covariance
        cov_f = fx.dot(self.Wc).dot(fx.T) - np.outer(mean_f, mean_f.T)
        cov_f += self.model_var
        # input-output covariance
        dx = x - mean
        cov_fx = self.D.dot(dx).dot(self.Wcc).dot(fx.T)
        return mean_f, cov_f, cov_fx

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
        K = np.exp(2 * np.log(alpha) - 0.5 * self._maha(inp, inp))
        iK = cho_solve(cho_factor(K + jitter * eye_n), eye_n)
        B = iLam2 + eye_d  # (D, D)
        c = alpha ** 2 / np.sqrt(det(B))
        t = inp.dot(inv(B))  # inn*(P + Lambda)^-1
        l = np.exp(-0.5 * np.sum(inp * t, 1))  # (N, 1)
        zet = 2 * np.log(alpha) - 0.5 * np.sum(inp * inp, 1)
        inp = inp.dot(iLam1)
        R = 2 * iLam2 + eye_d
        t = 1 / np.sqrt(det(R))
        L = np.exp((zet[:, na] + zet[:, na].T) + self._maha(inp, -inp, V=0.5 * inv(R)))
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

    def _min_var_sigmas(self):
        # minimum variance point set
        # scipy.optimize Nelder-mead simplex method (no kernel derivatives w.r.t. sigmas needed)
        pass

    def _min_var_hypers(self):
        # finds hypers that minimize integral variance (these minimize MMD)
        # scipy.optimize has a lot of solvers available
        pass

    def _min_logmarglik_hypers(self):
        # finds hypers by maximizing the marginal likelihood (empirical bayes)
        # the multiple output dimensions should be reflected in the log marglik
        pass

    def _min_intvar_logmarglik_hypers(self):
        # finds hypers by minimizing the sum of log-marginal likelihood and the integral variance objectives
        pass

    @staticmethod
    def _maha(x, y, V=None):
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
        MD = (x2V[:, na] + y2V[:, na].T) - 2 * x.dot(V).dot(y.T)

        return MD
