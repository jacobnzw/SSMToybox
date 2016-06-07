import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import warnings
from numba import jit
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.stats import multivariate_normal

from transform import BayesianQuadratureTransform
from transforms.model import *


class GPQuad(BayesianQuadratureTransform):
    def __init__(self, dim, kernel, points, kern_hyp=None, point_par=None):
        super(GPQuad, self).__init__(dim, 'gp', kernel, points, kern_hyp, point_par)

    def _weights(self):
        x = self.model.points
        iK = self.model.kernel.eval_inv(x)
        # kernel expectations
        q = self.model.kernel.exp_x_kx(x)
        Q = self.model.kernel.exp_x_kxkx(x)
        R = self.model.kernel.exp_x_xkx(x)
        # quadrature weigts in terms of kernel expectations
        w_m = q.dot(iK)
        w_c = iK.dot(Q).dot(iK)
        w_cc = R.dot(iK)
        return w_m, w_c, w_cc

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

    def plot_tp_model(self, f, unit_sp, args, test_range=(-5, 5, 50), plot_dims=(0, 0)):
        # plot out_dim vs. in_dim
        in_dim, out_dim = plot_dims
        d, n = unit_sp.shape
        # test input must have the same dimension as specified in kernel
        test = np.linspace(*test_range)
        test_pts = np.zeros((d, len(test)))
        test_pts[in_dim, :] = test
        # function value observations at training points (unit sigma-points)
        y = np.apply_along_axis(f, 0, unit_sp, args)
        fx = np.apply_along_axis(f, 0, test_pts, args)  # function values at test points
        K = self.kern.K(unit_sp.T)  # covariances between sigma-points
        kx = self.kern.K(test_pts.T, unit_sp.T)  # covariance between test inputs and sigma-points
        kxx = self.kern.Kdiag(test_pts.T)  # prior predictive variance
        iK = cho_solve(cho_factor(K), np.eye(n))
        tp_scale = (self.nu - 2 + y.dot(iK).dot(y.T).squeeze()) / (self.nu - 2 + n)
        tp_mean = kx.dot(iK).dot(y[out_dim, :])  # TP mean
        tp_var = tp_scale * np.diag(np.diag(kxx) - kx.dot(iK).dot(kx.T))  # TP predictive variance
        # plot the TP mean, predictive variance and the true function
        plt.figure()
        plt.plot(test, fx[out_dim, :], color='r', ls='--', lw=2, label='true')
        plt.plot(test, tp_mean, color='b', ls='-', lw=2, label='TP mean')
        plt.fill_between(test, tp_mean + 2 * np.sqrt(tp_var), tp_mean - 2 * np.sqrt(tp_var),
                         color='b', alpha=0.25, label='TP variance')
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

    def _covariance(self, weights, fcn_evals, mean_out):
        scale = (self.nu - 2 + fcn_evals.dot(self.iK).dot(fcn_evals.T)) / (self.nu - 2 + self.n)
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + scale * self.model_var
