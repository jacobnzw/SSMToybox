import unittest

import matplotlib.pyplot as plt
import numpy as np

from transforms.bayesquad import GPQuad
from transforms.quad import Unscented


class GPQuadTest(unittest.TestCase):
    def test_weights_rbf(self):
        n = 1
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (n + kappa) - n
        unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        hypers = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((n,)), 'noise_var': 1e-8}
        tf = GPQuad(unit_sp, hypers)
        wm, wc, wcc = tf.weights_rbf()
        print 'wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc)
        print 'GP model variance: {}'.format(tf.model_var)

    def test_min_var_sigmas(self):
        d = 2
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (d + kappa) - d
        # unit_sp = Unscented.unit_sigma_points(n, np.sqrt(n + lam))
        unit_sp = np.random.rand(d, 5)
        hypers = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((d,)), 'noise_var': 1e-8}
        tf = GPQuad(unit_sp, hypers)
        # find minimum variance sigmas
        x_opt = tf._min_var_sigmas().reshape((tf.d, tf.n))
        # plot minvar sigmas
        # ax = plt.figure().add_axes([-2, -2, 2, 2])
        plt.scatter(x_opt[0, :], x_opt[1, :])
        plt.show()

    def test_min_var_hypers(self):
        d = 2
        kappa, alpha, beta = 0, 1.0, 2.0
        lam = alpha ** 2 * (d + kappa) - d
        unit_sp = Unscented.unit_sigma_points(d, np.sqrt(d + lam))
        # unit_sp = np.random.rand(d, 5)
        hypers = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones((d,)), 'noise_var': 1e-8}
        tf = GPQuad(unit_sp, hypers)
        # find minimum variance sigmas
        hyp_opt = tf._min_var_hypers()
        print "s2 = {0}\nel = {1}".format(hyp_opt[0], hyp_opt[1:])
