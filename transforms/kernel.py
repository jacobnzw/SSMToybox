import numpy as np


class Kernel:
    def __init__(self):
        pass

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
    def __init__(self):
        # init hypers, call super-init, etc.
        pass

    def eval(self):
        # separate method for k(x) and K? I think not.
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
