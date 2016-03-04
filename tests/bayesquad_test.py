import unittest

import numpy as np

from transforms.bayesquad import GPQuad
from transforms.quad import Unscented


class GaussianProcessTest(unittest.TestCase):
    def test_weights_rbf(self):
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        hypers = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((n, 1)), 'noise_var': 1e-8}
        tf = GPQuad(unit_sp, hypers)
        wm, wc, wcc = tf.weights_rbf()
        print 'wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc)
        print 'GP model variance: {}'.format(tf.model_var)
