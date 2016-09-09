from unittest import TestCase

import numpy as np
import numpy.linalg as la

from transforms.bayesquad import GPQ
from models.pendulum import Pendulum
from models.ungm import UNGM

np.set_printoptions(precision=4)


class GPQuadTest(TestCase):
    models = [UNGM, Pendulum]

    def test_weights_rbf(self):
        dim = 1
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(dim, )}
        # phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = GPQ(dim, 'rbf', 'ut', khyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        print('wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc))
        self.assertTrue(np.allclose(wc, wc.T), "Covariance weight matrix not symmetric.")
        wc = 0.5 * (wc + wc.T)
        self.assertTrue(np.array_equal(wc, wc.T))
        # print 'GP model variance: {}'.format(tf.model.exp_model_variance())

    def test_weights_rbf_alpha_invariance(self):
        dim = 5
        tf = GPQ(dim, 'rbf', 'ut')
        w0 = tf._weights([1] + dim * [1000])
        w1 = tf._weights([358.0] + dim * [1000.0])
        self.assertTrue(np.alltrue([np.array_equal(a, b) for a, b in zip(w0, w1)]))

    def test_apply(self):
        for ssm in self.models:
            f = ssm().dyn_eval
            dim = ssm.xD
            tf = GPQ(dim, 'rbf', 'ut')
            mean, cov = np.zeros(dim, ), np.eye(dim)
            tmean, tcov, tccov = tf.apply(f, mean, cov, np.atleast_1d(1.0))
            self.assertTrue(np.array_equal(tcov, tcov.T))
            la.cholesky(tcov)
            print("Transformed moments\nmean: {}\ncov: {}\nccov: {}".format(tmean, tcov, tccov))
