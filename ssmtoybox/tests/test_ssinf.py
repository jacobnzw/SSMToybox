from unittest import TestCase

import numpy as np
import numpy.linalg as la

from ssmtoybox.ssinf import GPQMKalman, BayesSardKalman, TPQStudent
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement, Pendulum2DTransition, Pendulum2DMeasurement
from ssmtoybox.utils import GaussRV, StudentRV


class BSQKalmanTest(TestCase):

    def setUpClass(self):
        # setup UNGM
        x0 = GaussRV(1)
        q = GaussRV(1, cov=np.array([[10.0]]))
        r = GaussRV(1)
        self.ungm_dyn = UNGMTransition(x0, q)
        self.ungm_obs = UNGMMeasurement(r)
        ungm_x = self.ungm_dyn.simulate_discrete(100)
        self.ungm_y = self.ungm_obs.simulate_measurements(ungm_x)

        # setup 2D pendulum
        x0 = GaussRV(2, mean=np.array([1.5, 0]), cov=0.01 * np.eye(2))
        dt = 0.01
        q = GaussRV(2, cov=0.01 * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]]))
        r = GaussRV(1, cov=np.array([[0.1]]))
        self.pend_dyn = Pendulum2DTransition(x0, q, dt=dt)
        self.pend_obs = Pendulum2DMeasurement(r)
        x = self.pend_dyn.simulate_discrete(100)
        self.pend_y = self.pend_obs.simulate_measurements(x)

    def test_init(self):
        kpar = np.array([[1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 2]])
        BayesSardKalman(self.ungm_dyn, self.ungm_obs, kpar, kpar)

    def test_filtering_ungm(self):
        kpar = np.array([[1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 2]])
        alg = BayesSardKalman(self.ungm_dyn, self.ungm_obs, kpar, kpar)
        alg.forward_pass(self.ungm_y[..., 0])

    def test_filtering_pendulum(self):
        kpar = np.array([[1, 1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 0, 2, 0],
                          [0, 0, 1, 0, 2]])
        alg = BayesSardKalman(self.pend_dyn, self.pend_obs, kpar, kpar)
        alg.forward_pass(self.pend_y[..., 0])


class TPQStudentTest(TestCase):

    def test_ungm_filtering(self):
        # setup UNGM with Student RVs
        x0 = StudentRV(1)
        q = StudentRV(1, scale=np.array([[10.0]]))
        dyn = UNGMTransition(x0, q)
        r = StudentRV(1)
        obs = UNGMMeasurement(r)
        # simulate data
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)

        kerpar = np.array([[1.0, 1.0]])
        filt = TPQStudent(dyn, obs, kerpar, kerpar)
        filt.forward_pass(y[..., 0])

    def test_tracking_filtering(self):
        # some higher dim tracking
        pass


class GPQMarginalizedTest(TestCase):
    def test_init(self):
        ssm = UNGMGaussSSM()
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')

    def test_time_update(self):
        ssm = UNGMGaussSSM()
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        alg._time_update(1)
        par_dyn, par_obs = np.array([1, 1]), np.array([1, 1])
        alg._time_update(1, par_dyn, par_obs)

    def test_laplace_approx(self):
        ssm = UNGMGaussSSM()
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        # Random measurement
        y = np.sqrt(10)*np.random.randn(1)
        alg._param_posterior_moments(y, 10)
        la.cholesky(alg.param_cov)

    def test_measurement_update(self):
        ssm = UNGMGaussSSM()
        ssm_state, ssm_observations = ssm.simulate(5)
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        alg._measurement_update(ssm_observations[:, 0, 0], 1)

    def test_filtering_ungm(self):
        ssm = UNGMGaussSSM()
        ssm_state, ssm_observations = ssm.simulate(100)
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        alg.forward_pass(ssm_observations[..., 0])

    def test_filtering_pendulum(self):
        ssm = PendulumGaussSSM()
        ssm_state, ssm_observations = ssm.simulate(100)
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        alg.forward_pass(ssm_observations[..., 0])