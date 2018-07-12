import unittest

from ssmtoybox.utils import *


class TestMetrics(unittest.TestCase):

    def setUp(self):
        dim = 5
        self.x = np.random.randn(dim, )
        self.m = np.random.randn(dim, )
        self.cov = np.random.randn(dim, dim)
        self.cov = self.cov.dot(self.cov.T)
        self.mse = np.random.randn(dim, dim)

    def test_nll(self):
        neg_log_likelihood(self.x, self.m, self.cov)

    def test_log_cred_ratio(self):
        log_cred_ratio(self.x, self.m, self.cov, self.mse)


class TestMSEMatrix(unittest.TestCase):

    def test_sample_mse_matrix(self):
        dim = 5
        mc = 100
        x = np.random.randn(dim, mc)
        m = np.random.randn(dim, mc)
        mse_matrix(x, m)
