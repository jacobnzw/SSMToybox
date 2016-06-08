from unittest import TestCase

import numpy as np

from bayesquad import GPQuad
from models.ungm import UNGM

np.set_printoptions(precision=4)


class GPQuadTest(TestCase):
    def test_weights_rbf(self):
        dim = 1
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(dim, )}
        # phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = GPQuad(dim, 'rbf', 'ut', khyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        print 'wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc)
        self.assertTrue(np.allclose(wc, wc.T), "Covariance weight matrix not symmetric.")
        wc = 0.5 * (wc + wc.T)
        self.assertTrue(np.array_equal(wc, wc.T))
        # print 'GP model variance: {}'.format(tf.model.exp_model_variance())

    def test_apply(self):
        dim = 1
        tf = GPQuad(dim, 'rbf', 'ut')
        f = UNGM().dyn_eval
        mean, cov = np.zeros(dim, ), np.eye(dim)
        tmean, tcov, tccov = tf.apply(f, mean, cov, np.atleast_1d(1.0))
        print "Transformed moments\nmean: {}\ncov: {}\nccov: {}".format(tmean, tcov, tccov)
