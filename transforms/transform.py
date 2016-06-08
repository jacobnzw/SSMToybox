from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na
from numpy.linalg import cholesky


# Causes import hell through circular dependencies: Model needs to import from quad to get to the points,
# but all transforms in quad need MomentTransform to inherit from it. If the statement is moved after the definition of
# MomentTransform and SigmaPointTransform classes then no circular dependencies occur.
# from model import GaussianProcess, StudentTProcess


class MomentTransform(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, f, mean, cov, pars):
        raise NotImplementedError


class SigmaPointTransform(MomentTransform):
    __metaclass__ = ABCMeta

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
    __metaclass__ = ABCMeta

    _supported_models_ = ['gp', 'tp']  # mgp, gpder, ...

    def __init__(self, dim, model='gp', kernel=None, points=None, kern_hyp=None, point_par=None):
        self.model = BayesianQuadratureTransform._get_model(dim, model, kernel, points, kern_hyp, point_par)
        self.d, self.n = self.model.points.shape
        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self._weights()

    def apply(self, f, mean, cov, pars):
        mean = mean[:, na]
        chol_cov = cholesky(cov)
        x = mean + chol_cov.dot(self.unit_sp)
        fx = self._fcn_eval(f, x, pars)
        mean_f = self._mean(self.wm, fx)
        cov_f = self._covariance(self.Wc, fx, mean_f)
        cov_fx = self._cross_covariance(self.Wcc, fx, chol_cov)
        return mean_f, cov_f, cov_fx

    @staticmethod
    def _get_model(dim, model, kernel, points, hypers, point_pars):
        from model import GaussianProcess, StudentTProcess  # import must be after SigmaPointTransform
        model = model.lower()
        # make sure kernel is supported
        if model not in BayesianQuadratureTransform._supported_models_:
            print 'Model {} not supported. Supported models are {}.'.format(kernel, Model._supported_kernels_)
            return None
        # initialize the chosen kernel
        if model == 'gp':
            return GaussianProcess(dim, kernel, points, hypers, point_pars)
        elif model == 'tp':
            return StudentTProcess(dim, kernel, points, hypers, point_pars)

    # TODO: specify requirements for shape of input/output for all of these fcns

    def minimum_variance_points(self, x0, hypers):
        # run optimizer to find minvar point sets using initial guess x0; requires implemented _integral_variance()
        pass

    @abstractmethod
    def _weights(self):
        # no need for input args because points and hypers are in self.model.points and self.model.kernel.hypers
        raise NotImplementedError

    @abstractmethod
    def _integral_variance(self, points, hypers):
        # can serve for finding minimum variance point sets or hyper-parameters
        # optimizers require the first argument to be the variable, a decorator could be used to interchange the first
        # two arguments, so that we don't have to define the same function twice only w/ different signature
        raise NotImplementedError

    @abstractmethod
    def _fcn_eval(self, fcn, x, fcn_pars):
        # derived class decides whether to return derivatives also
        raise NotImplementedError

    def _mean(self, weights, fcn_evals):
        return fcn_evals.dot(weights)

    def _covariance(self, weights, fcn_evals, mean_out):
        expected_model_var = self.model.exp_model_variance(fcn_evals)
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _cross_covariance(self, weights, fcn_evals, chol_cov_in):
        return fcn_evals.dot(weights.T).dot(chol_cov_in.T)
