from unittest import TestCase

import numpy as np
import numpy.linalg as la

from ssmtoybox.ssinf import GPQMKalman, BayesSardKalman, TPQStudent
from ssmtoybox.ssinf import UnscentedKalman
from ssmtoybox.ssmod import UNGMTransition, Pendulum2DTransition, CoordinatedTurnTransition
from ssmtoybox.ssmod import UNGMMeasurement, Pendulum2DMeasurement, BearingMeasurement
from ssmtoybox.utils import GaussRV, StudentRV


class BSQKalmanTest(TestCase):

    @classmethod
    def setUpClass(cls):
        # setup UNGM
        x0 = GaussRV(1)
        q = GaussRV(1, cov=np.array([[10.0]]))
        r = GaussRV(1)
        cls.ungm_dyn = UNGMTransition(x0, q)
        cls.ungm_obs = UNGMMeasurement(r, cls.ungm_dyn.dim_out)
        ungm_x = cls.ungm_dyn.simulate_discrete(100)
        cls.ungm_y = cls.ungm_obs.simulate_measurements(ungm_x)

        # setup 2D pendulum
        x0 = GaussRV(2, mean=np.array([1.5, 0]), cov=0.01 * np.eye(2))
        dt = 0.01
        q = GaussRV(2, cov=0.01 * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]]))
        r = GaussRV(1, cov=np.array([[0.1]]))
        cls.pend_dyn = Pendulum2DTransition(x0, q, dt=dt)
        cls.pend_obs = Pendulum2DMeasurement(r, cls.pend_dyn.dim_out)
        x = cls.pend_dyn.simulate_discrete(100)
        cls.pend_y = cls.pend_obs.simulate_measurements(x)

        # setup coordinated turn bearing only tracking
        m0 = np.array([1000, 300, 1000, 0, np.deg2rad(-3.0)])
        P0 = np.diag([100, 10, 100, 10, 0.1])
        x0 = GaussRV(5, m0, P0)
        dt = 0.1
        rho_1, rho_2 = 0.1, 1.75e-4
        A = np.array([[dt**3/3, dt**2/2],
                      [dt**2/2, dt]])
        Q = np.zeros((5, 5))
        Q[:2, :2], Q[2:4, 2:4], Q[4, 4] = rho_1*A, rho_1*A, rho_2*dt
        q = GaussRV(5, cov=Q)
        r = GaussRV(4, cov=10e-3*np.eye(4))
        sen = np.vstack((1000 * np.eye(2), -1000 * np.eye(2))).astype(np.float)
        cls.ctb_dyn = CoordinatedTurnTransition(x0, q)
        cls.ctb_obs = BearingMeasurement(r, 5, state_index=[0, 2], sensor_pos=sen)
        x = cls.ctb_dyn.simulate_discrete(100)
        cls.ctb_y = cls.ctb_obs.simulate_measurements(x)

    def test_init(self):
        kpar = np.array([[1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 2]])
        BayesSardKalman(self.ungm_dyn, self.ungm_obs, kpar, kpar, alpha, alpha)

    def test_filtering_ungm(self):
        kpar = np.array([[1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 2]])
        alg = BayesSardKalman(self.ungm_dyn, self.ungm_obs, kpar, kpar, alpha, alpha)
        alg.forward_pass(self.ungm_y[..., 0])

    def test_filtering_pendulum(self):
        kpar = np.array([[1, 1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 0, 2, 0],
                          [0, 0, 1, 0, 2]])
        alg = BayesSardKalman(self.pend_dyn, self.pend_obs, kpar, kpar, alpha, alpha)
        alg.forward_pass(self.pend_y[..., 0])

    def test_filtering_ct_bearing(self):
        kpar = np.array([[1, 100, 100, 100, 100, 1]], dtype=np.float)
        alpha = np.hstack((np.eye(5), 2*np.eye(5))).astype(int)
        alg = BayesSardKalman(self.ctb_dyn, self.ctb_obs, kpar, kpar, alpha, alpha)
        alg.tf_dyn.model.model_var = 0 #np.diag([0.2])
        alg.tf_meas.model.model_var = 0
        # alg = UnscentedKalman(self.ctb_dyn, self.ctb_obs)
        alg.forward_pass(self.ctb_y[..., 0])


class TPQStudentTest(TestCase):

    def test_ungm_filtering(self):
        # setup UNGM with Student RVs
        x0 = StudentRV(1)
        q = StudentRV(1, scale=np.array([[10.0]]))
        r = StudentRV(1)
        dyn = UNGMTransition(x0, q)
        obs = UNGMMeasurement(r, dyn.dim_out)
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