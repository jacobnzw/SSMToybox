import numpy as np
import numpy.linalg as la
from numpy import newaxis as na


class Kernel(object):
    # list of strings of supported hyperparameters
    _hyperparameters_ = None

    def __init__(self, dim, hypers):
        self.dim = dim
        # use default hypers if unspecified
        self.hypers = self._get_default_hyperparameters(dim) if hypers is None else hypers

    # evaluation
    def eval(self, x1, x2=None):
        raise NotImplementedError

    # expectations
    def exp_x_kx(self, x):
        raise NotImplementedError

    def exp_x_xkx(self, x):
        raise NotImplementedError

    def exp_x_kxx(self):
        raise NotImplementedError

    def exp_xy_kxy(self):
        raise NotImplementedError

    def exp_x_kxkx(self, x):
        raise NotImplementedError

    # derivatives

    def _get_default_hyperparameters(self, dim):
        raise NotImplementedError


class RBF(Kernel):
    _hyperparameters_ = ['alpha', 'el']

    def __init__(self, dim, hypers=None, jitter=1e-8):
        super(RBF, self).__init__(dim, hypers)
        self.alpha = hypers['alpha']
        self.el = hypers['el']
        self.jitter = jitter
        # pre-computation for convenience
        self.lam = np.diag(self.el ** 2)
        self.inv_lam = np.diag(self.el ** -2)
        self.sqrt_inv_lam = np.diag(np.sqrt(self.el ** -1))
        self.eye_d = np.eye(dim)

    def eval(self, x1, x2=None):
        # ensure correct dimensions of x1, x2
        if x2 is None:
            x2 = x1
        x1 = self.sqrt_inv_lam.dot(x1)
        x2 = self.sqrt_inv_lam.dot(x2)
        return np.exp(2 * np.log(self.alpha) - 0.5 * self._maha(x2.T, x1.T))

    def exp_x_kx(self, x):
        # a.k.a. kernel mean map w.r.t. standard Gaussian PDF
        c = self.alpha ** 2 * (la.det(self.inv_lam + self.eye_d)) ** -0.5
        return c * self._maha(x, x, V=0.5 * la.inv(self.lam + self.eye_d))

    def exp_x_xkx(self, x):
        pass

    def exp_x_kxx(self):
        pass

    def exp_xy_kxy(self):
        pass

    def exp_x_kxkx(self, x):
        pass

    def _get_default_hyperparameters(self, dim):
        return {'alpha': 1.0, 'el': 1.0 * np.ones(dim, )}

    def _maha(self, x, y, V=None):
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


class Affine(Kernel):
    def __init__(self, dim):
        pass

    def eval(self):
        pass

    def exp_x_kx(self):
        pass

    def exp_x_xkx(self):
        pass

    def exp_x_kxx(self):
        pass

    def exp_xy_kxy(self):
        pass

    def exp_x_kxkx(self):
        pass
