from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from kernel import RBF
from quad import SphericalRadial, Unscented, GaussHermite


class Model(object):

    __metaclass__ = ABCMeta

    _supported_points_ = ['sr', 'ut', 'gh']
    _supported_kernels_ = ['rbf']

    def __init__(self, dim, kernel, points, kern_hyp=None, point_hyp=None):
        if kern_hyp is None:
            kern_hyp = {}
        if point_hyp is None:
            point_hyp = {}
        # init kernel and sigma-points
        self.kernel = Model.get_kernel(dim, kernel, kern_hyp)
        self.points = Model.get_points(dim, points, point_hyp)
        # save for printing
        self.str_pts = points
        self.str_pts_hyp = str(point_hyp)
        # may no longer be necessary now that jitter is in kernel
        self.d, self.n = self.points.shape
        self.eye_d, self.eye_n = np.eye(self.d), np.eye(self.n)

    def __str__(self):
        return '{}\n{} {}'.format(self.kernel, self.str_pts, self.str_pts_hyp)

    @abstractmethod
    def predict(self, test_data, fcn_obs):
        # model predictive mean and variance to be implemented by GP and TP classes
        raise NotImplementedError

    @abstractmethod
    def exp_model_variance(self, fcn_obs):
        # each model has to implement this using kernel expectations
        raise NotImplementedError

    @abstractmethod
    def marginal_log_likelihood(self, hypers, observations):
        # model specific marginal likelihood, will serve as an objective function passed into the optimizer
        raise NotImplementedError

    def optimize_hypers_max_ml(self, hypers_0, points, observations):
        # general routine minimizing negative marginal log-likelihood
        pass

    def plot_model(self, test_data, fcn_obs, fcn_true=None, in_dim=0):
        # general plotting routine for all models defined in terms of model's predictive mean and variance
        assert in_dim <= self.d - 1

        fcn_obs = np.squeeze(fcn_obs)
        fcn_true = np.squeeze(fcn_true)
        # model predictive mean and variance
        mean, var = self.predict(test_data, fcn_obs)
        std = np.sqrt(var)
        test_data = np.squeeze(test_data[in_dim, :])
        # set plot title according to model
        fig_title = self.__class__.__name__ + ' model of the integrand'
        # plot training data, predictive mean and variance
        fig = plt.figure(fig_title)
        plt.fill_between(test_data, mean - 2 * std, mean + 2 * std, color='0.1', alpha=0.15)
        plt.plot(test_data, mean, color='k', lw=2)
        plt.plot(self.points[in_dim, :], fcn_obs, 'ko', ms=8)
        # true function values at test points if provided
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
            print 'Kernel {} not supported. Supported kernels are {}.'.format(kernel, Model._supported_kernels_)
            return None
        # initialize the chosen kernel
        if kernel == 'rbf':
            return RBF(dim, hypers)
        elif kernel == 'affine':
            return Affine(dim, hypers)


class GaussianProcess(Model):  # consider renaming to GaussianProcessRegression/GPRegression, same for TP
    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        super(GaussianProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)

    def predict(self, test_data, fcn_obs):
        iK = self.kernel.eval_inv(self.points)
        kx = self.kernel.eval(test_data, self.points)
        kxx = self.kernel.eval(test_data, test_data, diag=True)
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        return mean, var

    def exp_model_variance(self, fcn_obs):
        q_bar = self.kernel.exp_x_kxx()
        Q = self.kernel.exp_x_kxkx(self.points)
        iK = self.kernel.eval_inv(self.points)
        return q_bar - np.trace(Q.dot(iK))

    def marginal_log_likelihood(self, hypers, observations):
        pass


class StudentTProcess(Model):
    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None, nu=None):
        super(StudentTProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)
        nu = 3.0 if nu is None else nu
        assert nu > 2, 'Degrees of freedom (nu) must be > 2.'
        self.nu = nu

    def predict(self, test_data, fcn_obs, nu=None):
        if nu is None:
            nu = self.nu
        iK = self.kernel.eval_inv(self.points)
        kx = self.kernel.eval(test_data, self.points)
        kxx = self.kernel.eval(test_data, test_data, diag=True)
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        scale = (nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (nu - 2 + self.n)
        return mean, scale * var

    def exp_model_variance(self, fcn_obs):
        fcn_obs = np.squeeze(fcn_obs)
        q_bar = self.kernel.exp_x_kxx()
        Q = self.kernel.exp_x_kxkx(self.points)
        iK = self.kernel.eval_inv(self.points)
        scale = (self.nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (self.nu - 2 + self.n)
        return scale * (q_bar - np.trace(Q.dot(iK)))

    def marginal_log_likelihood(self, hypers, observations):
        pass
