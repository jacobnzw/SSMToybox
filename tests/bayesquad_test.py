from unittest import TestCase

import numpy as np
# from models.ungm import UNGM
# from models.pendulum import Pendulum
from bayesquad import GPQuad


# from transforms.quad import Unscented, GaussHermite
# import time


class GPQuadTest(TestCase):
    def test_weights_rbf(self):
        dim = 1
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(dim, )}
        tf = GPQuad(dim, 'rbf', 'ut', khyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        print 'wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc)
        # print 'GP model variance: {}'.format(tf.model.exp_model_variance())
