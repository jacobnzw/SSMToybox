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

        # TODO: make a parent class for BQ-based transform
