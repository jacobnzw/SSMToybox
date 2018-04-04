from unittest import TestCase

import numpy as np
import numpy.linalg as la

from ssinf import GPQMKalman
from models.ungm import UNGM
from models.pendulum import Pendulum


class GPQMarginalizedTest(TestCase):
    def test_init(self):
        ssm = UNGM()
        alg = GPQMKalman(ssm, 'rbf', 'sr')

    def test_time_update(self):
        ssm = UNGM()
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        alg._time_update(1)
        par_dyn, par_obs = np.array([1, 1]), np.array([1, 1])
        alg._time_update(1, par_dyn, par_obs)

    def test_laplace_approx(self):
        ssm = UNGM()
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        # Random measurement
        y = np.sqrt(10)*np.random.randn(1)
        alg._param_posterior_moments(y, 10)
        la.cholesky(alg.param_cov)

    def test_measurement_update(self):
        ssm = UNGM()
        ssm_state, ssm_observations = ssm.simulate(5)
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        alg._measurement_update(ssm_observations[:, 0, 0], 1)

    def test_filtering_ungm(self):
        ssm = UNGM()
        ssm_state, ssm_observations = ssm.simulate(100)
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        alg.forward_pass(ssm_observations[..., 0])

    def test_filtering_pendulum(self):
        ssm = Pendulum()
        ssm_state, ssm_observations = ssm.simulate(100)
        alg = GPQMKalman(ssm, 'rbf', 'sr')
        alg.forward_pass(ssm_observations[..., 0])