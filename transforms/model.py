import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from scipy.linalg import cho_factor, cho_solve
from quad import *
from kernel import *


class Model(object):
    __metaclass__ = ABCMeta
    _supported_points_ = ['sr', 'ut', 'gh']
    _supported_kernels_ = ['rbf']

    def __init__(self, dim, kernel, points, kern_hyp=None, point_hyp=None):
        if kern_hyp is None:
            kern_hyp = {}
        if point_hyp is None:
            point_hyp = {}
        self.jitter = 1e-8
        # init kernel and sigma-points
        self.kernel = Model.get_kernel(dim, kernel, kern_hyp)
        self.points = Model.get_points(dim, points, point_hyp)
        self.d, self.n = self.points.shape
        self.eye_d, self.eye_n = np.eye(self.d), np.eye(self.n)

    @abstractmethod
    def predict(self, test_data, fcn_obs):
        # model predictive mean and variance to be implemented by GP and TP classes
        raise NotImplementedError

    @abstractmethod
    def marginal_log_likelihood(self, hypers, observations):
        # model specific marginal likelihood, will serve as an objective function passed into the optimizer
        raise NotImplementedError

    def optimize_hypers_max_ml(self, hypers_0, points, observations):
        # general routine minimizing the general negative marginal log-likelihood
        pass

    def plot_model(self, test_data, fcn_obs, fcn_true=None, in_dim=0):
        # general plotting routine for all models defined in terms of model's predictive mean and variance

        fcn_obs = np.squeeze(fcn_obs)
        fcn_true = np.squeeze(fcn_true)
        # model predictive mean and variance
        mean, var = self.predict(test_data, fcn_obs)
        std = np.sqrt(var)
        test_data = np.squeeze(test_data[in_dim, :])
        # set title according to model
        fig_title = self.__class__.__name__ + ' model of the integrand'
        # plot training data, predictive mean and variance
        fig = plt.figure(fig_title)
        plt.fill_between(test_data, mean - 2 * std, mean + 2 * std, color='0.1', alpha=0.15)
        plt.plot(test_data, mean, color='k', lw=2)
        plt.plot(self.points[in_dim, :], fcn_obs, 'ko', ms=8)
        if fcn_true is not None:
            plt.plot(test_data, fcn_true, lw=2, ls='--', color='tomato')
        plt.show()

    @staticmethod
    def get_points(dim, points, kwargs):
        points = points.lower()
        # make sure points is supported
        if points not in Model._supported_points_:
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
    def get_kernel(dim, kernel, hypers):
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

    def _cho_inv(self, A, b):
        return cho_solve(cho_factor(A), b)


class GaussianProcess(Model):  # consider renaming to GaussianProcessRegression/GPRegression, same for TP
    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        super(GaussianProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)

    def predict(self, test_data, fcn_obs):
        iK = self._cho_inv(self.kernel.eval(self.points) + self.jitter * self.eye_n, self.eye_n)
        kx = self.kernel.eval(test_data, self.points)
        kxx = self.kernel.eval(test_data, test_data, diag=True)
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        return mean, var

    def marginal_log_likelihood(self, hypers, observations):
        pass


class StudentTProcess(Model):
    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        super(StudentTProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)

    def predict(self, test_data, fcn_obs, nu=3.0):
        iK = self._cho_inv(self.kernel.eval(self.points) + self.jitter * self.eye_n, self.eye_n)
        kx = self.kernel.eval(test_data, self.points)
        kxx = self.kernel.eval(test_data, test_data, diag=True)
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        scale = (nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (nu - 2 + self.n)
        return mean, scale * var

    def marginal_log_likelihood(self, hypers, observations):
        pass
