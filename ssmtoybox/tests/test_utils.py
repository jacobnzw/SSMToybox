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


class TestGaussMixture(unittest.TestCase):

    def test_gauss_mixture(self):
        means = ([0, 0], [1, 1], [3, 0], [0, -3])
        covs = (0.1 * np.eye(2), 0.2 * np.eye(2), 0.3 * np.eye(2), 0.1 * np.eye(2))
        alphas = (0.15, 0.3, 0.4, 0.15)
        num_samples = 1000
        samples, indexes = gauss_mixture(means, covs, alphas, num_samples, return_indices=True)

        import matplotlib.pyplot as plt
        plot_opts = {'linestyle': '', 'marker': '.', 'markersize': 2}
        for i in range(len(alphas)):
            sel = indexes == i
            plt.plot(samples[sel, 0], samples[sel, 1], **plot_opts)
        plt.show()

    def test_indices_switch(self):
        dim = 2
        means = ([0, 0], [1, 1], [3, 0], [0, -3])
        covs = (0.1 * np.eye(dim), 0.2 * np.eye(dim), 0.3 * np.eye(2), 0.1 * np.eye(dim))
        alphas = (0.15, 0.3, 0.4, 0.15)
        num_samples = 1000

        samples, indexes = gauss_mixture(means, covs, alphas, num_samples, return_indices=True)

        self.assertEqual((num_samples, dim), samples.shape)
        self.assertEqual((num_samples, ), indexes.shape)

        samples = gauss_mixture(means, covs, alphas, num_samples, return_indices=False)
        self.assertEqual(samples.shape, (num_samples, dim))

    def test_size(self):
        dim = 2
        means = ([0, 0], [1, 1], [3, 0], [0, -3])
        covs = (0.1 * np.eye(dim), 0.2 * np.eye(dim), 0.3 * np.eye(2), 0.1 * np.eye(dim))
        alphas = (0.15, 0.3, 0.4, 0.15)
        num_samples = 1000
        mc_sims = 10

        samples, indexes = gauss_mixture(means, covs, alphas, (num_samples, mc_sims), return_indices=True)

        self.assertEqual((num_samples, mc_sims, dim), samples.shape)
        self.assertEqual((num_samples, mc_sims), indexes.shape)
