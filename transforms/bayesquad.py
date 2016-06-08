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


class GPQuad(BayesianQuadratureTransform):  # consider renaming to GPQTransform
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


class TPQuad(BayesianQuadratureTransform):
    def __init__(self, dim, unit_sp=None, hypers=None, nu=3.0):
        super(TPQuad, self).__init__(dim, 'tp', kernel, points, kern_hyp, point_par)

    def _weights(self, sigma_points, hypers):
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
