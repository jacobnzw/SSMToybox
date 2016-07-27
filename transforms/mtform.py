

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na
from numpy.linalg import cholesky


# Causes import hell through circular dependencies: Model needs to import from quad to get to the points,
# but all transforms in quad need MomentTransform to inherit from it. If the statement is moved after the definition of
# MomentTransform and SigmaPointTransform classes then no circular dependencies occur.
# from model import GaussianProcess, StudentTProcess


# TODO: documentation


class MomentTransform(object, metaclass=ABCMeta):
    @abstractmethod
    def apply(self, f, mean, cov, pars):
        raise NotImplementedError


class SigmaPointTransform(MomentTransform, metaclass=ABCMeta):
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


class SigmaPointTruncTransform(SigmaPointTransform):
    # sigma-point transform respecting effective input dimensionality
    # TODO: cross-covariance can still be computed with the lower-dimensional rule if lowdim points are extended with
    # zeros to match the state dimension, amounts to updating only the observed state dimensions
    # TODO: test again

    def apply(self, f, mean, cov, pars):
        mean = mean[:, na]

        # consider only effective dimension
        mean_eff = mean[:self.dim_eff]
        cov_eff = cov[:self.dim_eff, :self.dim_eff]

        # form sigma-points from unit sigma-points
        x_eff = mean_eff + cholesky(cov_eff).dot(self.unit_sp_eff)
        x = mean + cholesky(cov).dot(self.unit_sp)

        # push sigma-points through non-linearity
        fx_eff = np.apply_along_axis(f, 0, x_eff, pars)
        fx = np.apply_along_axis(f, 0, x, pars)

        # output mean
        mean_f = fx_eff.dot(self.wm)
        # output covariance
        dfx_eff = fx_eff - mean_f[:, na]
        dfx = fx - mean_f[:, na]
        cov_f = dfx_eff.dot(self.Wc).dot(dfx_eff.T)
        # input-output covariance
        cov_fx = dfx_eff.dot(self.Wcc).dot((x - mean).T)
        return mean_f, cov_f, cov_fx
