from unittest import TestCase

import numpy as np
from numpy import newaxis as na

from transforms.model import GaussianProcess

# fcn = lambda x: np.sin((x + 1) ** -1)
fcn = lambda x: 0.5 * x + 25 * x / (1 + x ** 2)


# fcn = lambda x: 0.05*x ** 2
# fcn = lambda x: x


class GPModelTest(TestCase):
    # TODO: could be general test class for any model

    def test_init(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        phyp = {'alpha': 1.0}
        GaussianProcess(1)
        GaussianProcess(1, kernel='rbf', points='ut', kern_hyp=khyp, point_hyp=phyp)

    def test_plotting(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        model.plot_model(xtest, y, fcn_true=f)

    def test_exp_model_variance(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        y = fcn(model.points)
        self.assertTrue(model.exp_model_variance(y) >= 0)

    def test_log_marginal_likelihood(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        y = fcn(model.points)
        lhyp = np.log([1.0, 3.0])
        f, df = model.neg_log_marginal_likelihood(lhyp, y.T)

    def test_hypers_optim(self):
        khyp = {'alpha': 1.0, 'el': 1.0 * np.ones(1)}
        model = GaussianProcess(1, points='gh', kern_hyp=khyp, point_hyp={'degree': 3})
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        # plot before optimization
        model.plot_model(xtest, y, fcn_true=f)
        lhyp0 = np.log([1.0, 1.0])
        b = ((np.log(0.9), np.log(1.1)), (None, None))
        opt_result = model.optimize_ml(lhyp0, y.T, method='BFGS', jac=False, bounds=b)
        hyp_opt = np.exp(opt_result.x)
        print opt_result
        print 'ML-II hypers: alpha = {:.4f}, el = {:.4f} '.format(hyp_opt[0], hyp_opt[1])
        # plot after optimization
        model.plot_model(xtest, y, fcn_true=f, hyp=hyp_opt)

        # TODO: test fitting of multioutput GPs, GPy supports this in GPRegression
        # plot NLML surface
        # x = np.log(np.mgrid[1:10:0.5, 0.5:20:0.5])
        # m, n = x.shape[1:]
        # z = np.zeros(x.shape[1:])
        # for i in range(m):
        #     for j in range(n):
        #         z[i, j], grad = model.neg_log_marginal_likelihood(x[:, i, j], y.T)
        #
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface((x[0, ...]), (x[1, ...]), z, linewidth=0.5, alpha=0.5, rstride=2, cstride=2)
        # ax.set_xlabel('alpha')
        # ax.set_ylabel('el')
        # plt.show()

    def test_hypers_optim_regularized(self):
        khyp = {'alpha': 1.0, 'el': 1.0 * np.ones(1)}
        model = GaussianProcess(1, points='gh', kern_hyp=khyp, point_hyp={'degree': 3})
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        # plot before optimization
        model.plot_model(xtest, y, fcn_true=f)
        lhyp0 = np.log([1.0, 1.0])
        b = ((np.log(0.9), np.log(1.1)), (None, None))
        opt_result = model.optimize_ml_regularized(lhyp0, y.T, method='BFGS', jac=False, bounds=b)
        hyp_opt = np.exp(opt_result.x)
        print opt_result
        print 'ML-II-REG hypers: alpha = {:.4f}, el = {:.4f} '.format(hyp_opt[0], hyp_opt[1])
        # plot after optimization
        model.plot_model(xtest, y, fcn_true=f, hyp=hyp_opt)


# class TPModelTest(TestCase):
#     def test_init(self):
#         khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
#         phyp = {'alpha': 1.0}
#         StudentTProcess(1)
#         StudentTProcess(1, kernel='rbf', points='ut', kern_hyp=khyp, point_hyp=phyp)
#
#     def test_plotting(self):
#         dim = 1
#         khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(dim, )}
#         model = StudentTProcess(dim, kern_hyp=khyp)
#         xtest = np.linspace(-5, 5, 50)[na, :]
#         y = fcn(model.points)
#         f = fcn(xtest)
#         model.plot_model(xtest, y, f)
#
#     def test_exp_model_variance(self):
#         khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
#         model = StudentTProcess(1, kern_hyp=khyp)
#         y = fcn(model.points)
#         self.assertTrue(model.exp_model_variance(y) >= 0)
