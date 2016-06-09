from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na
from numpy.linalg import cholesky


# Causes import hell through circular dependencies: Model needs to import from quad to get to the points,
# but all transforms in quad need MomentTransform to inherit from it. If the statement is moved after the definition of
# MomentTransform and SigmaPointTransform classes then no circular dependencies occur.
# from model import GaussianProcess, StudentTProcess


# TODO: documentation


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
