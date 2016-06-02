import numpy as np

from quad import Unscented, SphericalRadial, GaussHermite
from kernel import *


class Model:
    _supported_points_ = ['sr', 'ut', 'gh']
    _supported_kernels_ = ['rbf']

    def __init__(self, dim, kernel, points, kern_hyp=None, point_hyp=None):
        if kern_hyp is None:
            kern_hyp = {}
        if point_hyp is None:
            point_hyp = {}
        # init kernel
        self.kernel = Model.get_kernel(dim, kernel, **kern_hyp)
        # init points
        self.points = Model.get_points(dim, points, **point_hyp)

    @staticmethod
    def get_points(dim, points, **kwargs):
        points = points.lower()
        # make sure points is supported
        if points not in Model._supported_kernels_:
            print 'Points {} not supported. Supported points are {}.'.format(points, Model._supported_points_)
            return None
        # create chosen points
        if points == 'sr':
            return SphericalRadial.unit_sigma_points(dim)
        elif points == 'ut':
            return Unscented.unit_sigma_points(dim, **kwargs)
        elif points == 'gh':
            return GaussHermite.unit_sigma_points(dim, **kwargs)

    @staticmethod
    def get_kernel(dim, kernel, **kwargs):
        kernel = kernel.lower()
        # make sure kernel is supported
        if kernel not in Model._supported_kernels_:
            print 'Kernel {} not supported. Supported points are {}.'.format(kernel, Model._supported_kernels_)
            return None
        # initialize the chosen kernel
        if kernel == 'rbf':
            pass
        elif kernel == 'affine':
            pass
