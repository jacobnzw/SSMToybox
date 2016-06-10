from unittest import TestCase

import numpy as np
from numpy import newaxis as na

from transforms.model import *

fcn = lambda x: np.sin((x + 1) ** -1)


# fcn = lambda x: x ** 2
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
        model.plot_model(xtest, y, f)

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
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        y = fcn(model.points)
        lhyp0 = np.log([1.0, 1.0])
        lhyp_opt = model.optimize_hypers_max_ml(lhyp0, y.T)
        print lhyp_opt
        print 'ML-II hypers: {}'.format(np.exp(lhyp_opt.x))

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
