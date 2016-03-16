from __future__ import division

import numpy as np
from numpy import newaxis as na
from numpy.linalg import cholesky


class MomentTransform(object):
    def apply(self, f, mean, cov, pars):
        raise NotImplementedError


class SigmaPointTransform(MomentTransform):
    def apply(self, f, mean, cov, pars):
        mean = mean[:, na]
        # form sigma-points from unit sigma-points
        x = mean + cholesky(cov).dot(self.unit_sp)
        # push sigma-points through non-linearity
        fx = np.apply_along_axis(f, 0, x, pars)
        # output mean
        mean_f = fx.dot(self.wm)
        # output covariance
        dfx = fx - mean_f[:, na]
        cov_f = dfx.dot(self.Wc).dot(dfx.T)
        # input-output covariance
        cov_fx = dfx.dot(self.Wc).dot((x - mean).T)
        return mean_f, cov_f, cov_fx


class BayesianQuadratureTransform(MomentTransform):
    # it's possible I'll need to make some more specialized parent classes
    def __init__(self, dim, unit_sp=None, hypers=None):
        # use default sigma-points if not provided
        self.unit_sp = unit_sp if unit_sp is not None else self.default_sigma_points(dim)  # (d, n)
        # get number of sigmas (n) and dimension of sigmas (d)
        self.d, self.n = self.unit_sp.shape
        assert self.d == dim  # check unit sigmas have proper dimension
        # use default kernel hyper-parameters if not provided
        self.hypers = hypers if hypers is not None else self.default_hypers(dim)
        # BQ weights given the unit sigma-points, kernel hyper-parameters and the kernel
        self.wm, self.Wc, self.Wcc = self._weights(self.unit_sp, self.hypers)

    def apply(self, f, mean, cov, pars):
        # method defined in terms of abstract private functions for computing mean, covariance and cross-covariance
        mean = mean[:, na]
        x = mean + cholesky(cov).dot(self.unit_sp)
        fx = self._fcn_eval(f, x, pars)
        mean_f = self._mean(self.wm, fx)
        cov_f = self._covariance(self.Wc, fx, mean_f)
        cov_fx = self._cross_covariance(self.Wcc, fx, x, mean_f, mean)
        return mean_f, cov_f, cov_fx

    def default_sigma_points(self, dim):
        # create unscented points
        c = np.sqrt(dim)
        return np.hstack((np.zeros((dim, 1)), c * np.eye(dim), -c * np.eye(dim)))

    def default_hypers(self, dim):
        # define default hypers
        return {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(dim, ), 'noise_var': 1e-8}

    def _weights(self, sigma_points, hypers):
        # implementation will differ based on kernel
        # it's possible it will call functions which implement weights for particular kernel
        raise NotImplementedError

    # TODO: specify requirements for shape of input/output for all of these fcns
    def _fcn_eval(self, fcn, x, fcn_pars):
        # derived class decides whether to return derivatives also
        raise NotImplementedError

    def _mean(self, weights, fcn_evals):
        raise NotImplementedError

    def _covariance(self, weights, fcn_evals, mean_out):
        raise NotImplementedError

    def _cross_covariance(self, weights, fcn_evals, x, mean_out, mean_in):
        raise NotImplementedError
