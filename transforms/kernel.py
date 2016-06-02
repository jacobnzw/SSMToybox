import numpy as np


class Kernel:
    # list of strings of supported hyperparameters
    _hyperparameters_ = None
    # dictionary of defaul values of hyperparameters
    _default_hyperparameters_ = None

    def __init__(self, dim, hypers):
        self.dim = dim
        # use default hypers if unspecified
        self.hypers = self._default_hyperparameters_ if hypers is None else hypers

    # evaluation
    def eval(self):
        raise NotImplementedError

    # expectations
    def exp_x_kx(self):
        raise NotImplementedError

    def exp_x_kxx(self):
        raise NotImplementedError

    def exp_xy_kxy(self):
        raise NotImplementedError

    def exp_x_kxkx(self):
        raise NotImplementedError
        # derivatives


class RBF(Kernel):
    _hyperparameters_ = ['alpha', 'el', 'jitter']
    _defaul_hyperparameters_ = {'alpha': 1.0, 'el': 1.0, 'jitter': 1e-8}

    def __init__(self, dim, hypers=None):
        super(RBF, self).__init__(dim, hypers)

    def eval(self, x1, x2=None):
        pass

    def exp_x_kx(self):
        pass

    def exp_x_kxx(self):
        pass

    def exp_xy_kxy(self):
        pass

    def exp_x_kxkx(self):
        pass


class Affine(Kernel):
    def __init__(self, dim):
        pass

    def eval(self):
        pass

    def exp_x_kx(self):
        pass

    def exp_x_kxx(self):
        pass

    def exp_xy_kxy(self):
        pass

    def exp_x_kxkx(self):
        pass
