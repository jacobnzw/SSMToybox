import numpy as np

from quad import *
from kernel import *


class Model(object):
    _supported_points_ = ['sr', 'ut', 'gh']
    _supported_kernels_ = ['rbf']

    def __init__(self, dim, kernel, points, kern_hyp=None, point_hyp=None):
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
    def get_kernel(dim, kernel, **hypers):
        kernel = kernel.lower()
        # make sure kernel is supported
        if kernel not in Model._supported_kernels_:
            print 'Kernel {} not supported. Supported points are {}.'.format(kernel, Model._supported_kernels_)
            return None
        # initialize the chosen kernel
        if kernel == 'rbf':
            return RBF(dim, hypers)
        elif kernel == 'affine':
            return Affine(dim, hypers)

    def predict_moments(self, test_data, points, observations):
        # model predictive mean and variance to be implemented by GP and TP classes
        raise NotImplementedError

    def marginal_log_likelihood(self, hypers, points, observations):
        # model specific marginal likelihood, will serve as an objective function passed into the optimizer
        raise NotImplementedError

    def optimize_hypers_max_ml(self, hypers_0, points, observations):
        # general routine minimizing the general negative marginal log-likelihood
        pass

    def plot_model(self, points, observations):
        # general plotting routine for all models defined in terms of model's predictive mean and variance
        pass


class GaussianProcess(Model):  # consider renaming to GaussianProcessRegression/GPRegression, same for TP
    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        super(GaussianProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)

    def predict_moments(self, test_data, points, observations):
        pass

    def marginal_log_likelihood(self, hypers, points, observations):
        pass


class StudentTProcess(Model):
    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        super(StudentTProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)

    def predict_moments(self, test_data, points, observations):
        pass

    def marginal_log_likelihood(self, hypers, points, observations):
        pass
