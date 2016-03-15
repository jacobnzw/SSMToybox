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
    # it's possible I'll need to make some more specialed parent classes
    def __init__(self, dim, unit_sp, hypers=None):
        # set
        self.unit_sp = unit_sp  # (d, n)
        # get number of sigmas (n) and dimension of sigmas (d)
        self.d, self.n = self.unit_sp.shape
        assert self.d == dim  # check unit sigmas have proper dimension
        # set kernel hyper-parameters (manually or some principled method)
        self.hypers = self._min_var_hypers() if hypers is None else hypers
        # BQ weights given the unit sigma-points and the kernel hyper-parameters
        self.wm, self.Wc, self.Wcc = self.weights_rbf()

    def default_sigma_points(self, dim):
        # create unscented points
        return Unscented.unit_sigma_points(dim, 2)

    def default_hypers(self, dim):
        # define default hypers
        return {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(dim, ), 'noise_var': 1e-8}

    def apply(self, f, mean, cov, pars):
        # unsure which implementation to choose
        # GPQ and TPQ might be put under one roof, GPQ+D however has different equations
        # I could defined abstract private functions for computing mean, covariance and cross-covariance and call them
        # from this method
        x = mean[:, na] + cholesky(cov).dot(self.unit_sp)
        fx = self._fcn_observations(f, x, pars)  # derived class decides whether to return derivatives also
        mean_f = self._mean(self.wm, fx)
        cov_f = self._covariance(self.Wc, fx, mean, mean_f)
        cov_fx = self._cross_covariance(self.Wcc, fx)
        return mean_f, cov_f, cov_fx
