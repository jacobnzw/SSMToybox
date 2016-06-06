import numpy as np
import numpy.linalg as la
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from quad import *
from kernel import *


class Model(object):
    _supported_points_ = ['sr', 'ut', 'gh']
    _supported_kernels_ = ['rbf']

    def __init__(self, dim, kernel, points, kern_hyp=None, point_hyp=None):
        # init kernel and sigma-points
        self.kernel = Model.get_kernel(dim, kernel, **kern_hyp)
        self.points = Model.get_points(dim, points, **point_hyp)
        self.d, self.n = self.points.shape
        self.jitter = 1e-8 * np.eye(self.n)

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

    def predict(self, test_data, fcn_obs):
        # model predictive mean and variance to be implemented by GP and TP classes
        raise NotImplementedError

    def marginal_log_likelihood(self, hypers, observations):
        # model specific marginal likelihood, will serve as an objective function passed into the optimizer
        raise NotImplementedError

    def optimize_hypers_max_ml(self, hypers_0, points, observations):
        # general routine minimizing the general negative marginal log-likelihood
        pass

    def plot_model(self, test_data, fcn_obs):
        # general plotting routine for all models defined in terms of model's predictive mean and variance
        # TODO: which in/out dimensions to plot
        mean, var = self.predict(test_data, fcn_obs)
        std = np.sqrt(var)
        # plot training data, predictive mean and variance
        fig = plt.figure()
        plt.plot(self.points, fcn_obs, 'ko')
        plt.plot(test_data, mean)
        plt.fill_between(test_data, mean + 2 * std, mean - 2 * std)
        plt.show()

    def _cho_inv(self, A, b):
        return cho_solve(cho_factor(A), b)


class GaussianProcess(Model):  # consider renaming to GaussianProcessRegression/GPRegression, same for TP
    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        super(GaussianProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)

    def predict(self, test_data, fcn_obs):
        K = self.kernel.eval(self.points)
        kx = self.kernel.eval(test_data, self.points)
        kxx = self.kernel.eval(test_data, test_data, diag=True)
        kx_iK = self._cho_inv(K + self.jitter, kx).T
        return kx_iK.dot(fcn_obs), kxx - kx_iK.dot(kx)

    def marginal_log_likelihood(self, hypers, observations):
        pass


class StudentTProcess(Model):
    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        super(StudentTProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)

    def predict(self, test_data, fcn_obs, nu=3.0):
        K = self.kernel.eval(self.points)
        kx = self.kernel.eval(test_data, self.points)
        kxx = self.kernel.eval(test_data, test_data, diag=True)
        kx_iK = self._cho_inv(K + self.jitter, kx).T
        scale = (nu - 2 + fcn_obs.T.dot(self._cho_inv(K + self.jitter, fcn_obs))) / (nu - 2 + self.n)
        return kx_iK.dot(fcn_obs), scale * (kxx - kx_iK.dot(kx))

    def marginal_log_likelihood(self, hypers, observations):
        pass
