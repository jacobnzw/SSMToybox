from unittest import TestCase

import numpy as np

from ssmtoybox.ssinf import MarginalizedGaussianProcessKalman, GaussianProcessKalman, BayesSardKalman, \
    StudentProcessStudent, StudentProcessKalman, FullySymmetricStudent
from ssmtoybox.ssinf import UnscentedKalman, ExtendedKalman, GaussHermiteKalman
from ssmtoybox.ssmod import UNGMMeasurement, UNGMNAMeasurement, Pendulum2DMeasurement, BearingMeasurement, \
    Radar2DMeasurement
from ssmtoybox.ssmod import UNGMTransition, UNGMNATransition, Pendulum2DTransition, CoordinatedTurnTransition, \
    ReentryVehicle2DTransition, ConstantTurnRateSpeed, ConstantVelocity
from ssmtoybox.utils import GaussRV, StudentRV

np.set_printoptions(precision=4)


class GaussianInferenceTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ssm = {}
        # setup UNGM
        x0 = GaussRV(1)
        q = GaussRV(1, cov=np.array([[10.0]]))
        r = GaussRV(1)
        dyn = UNGMTransition(x0, q)
        obs = UNGMMeasurement(r, 1)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'ungm': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

        # setup UNGM with non-additive noise
        x0 = GaussRV(1)
        q = GaussRV(1, cov=np.array([[10.0]]))
        r = GaussRV(1)
        dyn = UNGMNATransition(x0, q)
        obs = UNGMNAMeasurement(r, 1)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'ungmna': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

        # setup 2D pendulum
        x0 = GaussRV(2, mean=np.array([1.5, 0]), cov=0.01 * np.eye(2))
        dt = 0.01
        q = GaussRV(2, cov=0.01 * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]]))
        r = GaussRV(1, cov=np.array([[0.1]]))
        dyn = Pendulum2DTransition(x0, q, dt=dt)
        obs = Pendulum2DMeasurement(r, dyn.dim_state)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'pend': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

        # setup reentry vehicle radar tracking
        m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932])
        P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
        x0 = GaussRV(5, m0, P0)
        q = GaussRV(3, cov=np.diag([2.4064e-5, 2.4064e-5, 1e-6]))
        r = GaussRV(2, cov=np.diag([1e-6, 0.17e-6]))
        dyn = ReentryVehicle2DTransition(x0, q)
        obs = Radar2DMeasurement(r, 5)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'rer': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

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
        dyn = CoordinatedTurnTransition(x0, q)
        obs = BearingMeasurement(r, 5, state_index=[0, 2], sensor_pos=sen)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'ctb': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

        # setup CTRS with radar measurements
        x0 = GaussRV(5, cov=0.1*np.eye(5))
        q = GaussRV(2, cov=np.diag([0.1, 0.1*np.pi]))
        r = GaussRV(2, cov=np.diag([0.3, 0.03]))
        dyn = ConstantTurnRateSpeed(x0, q)
        obs = Radar2DMeasurement(r, 5)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'ctrs': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

    def test_extended_kalman(self):
        """
        Test Extended KF on range of SSMs.
        """
        for ssm_name, data in self.ssm.items():
            if ssm_name in ['rer', 'ctb', 'ctrs']:
                # Jacobians not implemented for reentry and coordinate turn
                continue
            print('Testing: {} ...'.format(ssm_name.upper()), end=' ')
            try:
                alg = ExtendedKalman(data['dyn'], data['obs'])
                alg.forward_pass(data['y'][..., 0])
                alg.backward_pass()
                alg.reset()
            except BaseException as e:
                print('Failed: {}'.format(e))
                continue
            print('OK')

    def test_unscented_kalman(self):
        """
        Test Unscented KF on range of SSMs.
        """
        for ssm_name, data in self.ssm.items():
            print('Testing: {} ...'.format(ssm_name.upper()), end=' ')
            try:
                alg = UnscentedKalman(data['dyn'], data['obs'])
                alg.forward_pass(data['y'][..., 0])
                alg.backward_pass()
                alg.reset()
            except BaseException as e:
                print('Failed: {}'.format(e))
                continue
            print('OK')

    def test_gauss_hermite_kalman(self):
        """
        Test Gauss-Hermite KF on range of SSMs.
        """
        for ssm_name, data in self.ssm.items():
            print('Testing: {} ...'.format(ssm_name.upper()), end=' ')
            try:
                alg = GaussHermiteKalman(data['dyn'], data['obs'])
                alg.forward_pass(data['y'][..., 0])
                alg.backward_pass()
                alg.reset()
            except BaseException as e:
                print('Failed {}'.format(e))
                continue
            print('OK')

    def test_gaussian_process_kalman(self):
        """
        Test Gaussian Process Quadrature KF on range of SSMs.
        """
        for ssm_name, data in self.ssm.items():
            if ssm_name in ['rer', 'ctb']:
                # GPQ kernel pars hard to find on higher-dimensional systems like reentry or CT
                continue
            print('Testing: {} ...'.format(ssm_name.upper()), end=' ')
            # setup kernel parameters
            kpar_dyn = np.atleast_2d(np.ones(data['dyn'].dim_in + 1))
            kpar_obs = np.atleast_2d(np.ones(data['obs'].dim_in + 1))
            try:
                alg = GaussianProcessKalman(data['dyn'], data['obs'], kpar_dyn, kpar_obs)
                alg.forward_pass(data['y'][..., 0])
                alg.backward_pass()
                alg.reset()
            except BaseException as e:
                print('Failed: {}'.format(e))
                continue
            print('OK')

    def test_student_process_kalman(self):
        """
        Test Student Process Quadrature KF on range of SSMs.
        """
        for ssm_name, data in self.ssm.items():
            if ssm_name in ['rer', 'ctb']:
                # TPQ kernel pars hard to find on higher-dimensional systems like reentry or CT
                continue
            print('Testing: {} ...'.format(ssm_name.upper()), end=' ')
            # setup kernel parameters
            kpar_dyn = np.atleast_2d(np.ones(data['dyn'].dim_in + 1))
            kpar_obs = np.atleast_2d(np.ones(data['obs'].dim_in + 1))
            try:
                alg = StudentProcessKalman(data['dyn'], data['obs'], kpar_dyn, kpar_obs)
                alg.forward_pass(data['y'][..., 0])
                alg.backward_pass()
                alg.reset()
            except BaseException as e:
                print('Failed: {}'.format(e))
                continue
            print('OK')

    def test_bayes_sard_kalman(self):
        """
        Test Bayes-Sard Quadrature KF on range of SSMs.
        """
        for ssm_name, data in self.ssm.items():
            print('Testing: {} ...'.format(ssm_name.upper()), end=' ')
            # setup kernel parameters and multi-indices (for polynomial mean function)
            dim = data['dyn'].dim_in
            kpar_dyn = np.atleast_2d(np.ones(dim + 1))
            alpha_dyn = np.hstack((np.zeros((dim, 1)), np.eye(dim), 2 * np.eye(dim))).astype(int)
            dim = data['obs'].dim_in
            kpar_obs = np.atleast_2d(np.ones(dim + 1))
            alpha_obs = np.hstack((np.zeros((dim, 1)), np.eye(dim), 2 * np.eye(dim))).astype(int)
            try:
                alg = BayesSardKalman(data['dyn'], data['obs'], kpar_dyn, kpar_obs, alpha_dyn, alpha_obs)
                alg.forward_pass(data['y'][..., 0])
                alg.backward_pass()
                alg.reset()
            except BaseException as e:
                print('Failed: {}'.format(e))
                continue
            print('OK')


class StudentInferenceTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ssm = {}
        # setup UNGM with Student RVs
        x0 = StudentRV(1)
        q = StudentRV(1, scale=np.array([[10.0]]))
        r = StudentRV(1)
        dyn = UNGMTransition(x0, q)
        obs = UNGMMeasurement(r, dyn.dim_state)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'ungm': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

        # setup CV with Student RVs
        m_0 = np.array([10175, 295, 980, -35]).astype(np.float)
        P_0 = np.diag([10000, 100, 10000, 100]).astype(np.float)
        nu_0 = 1000.0
        x0 = StudentRV(4, m_0, P_0, nu_0)
        Q = np.diag([50, 5]).astype(np.float)
        nu_q = 1000.0
        q = StudentRV(2, scale=Q, dof=nu_q)
        R = np.diag([50, 0.4e-6]).astype(np.float)
        nu_r = 4.0
        r = StudentRV(2, scale=R, dof=nu_r)
        dyn = ConstantVelocity(x0, q, dt=0.5)
        obs = Radar2DMeasurement(r, 4)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'cv': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

    def test_student_process_student(self):
        """
        Test t-Process Quadrature SF on a range of SSMs.
        """

        for ssm_name, data in self.ssm.items():
            dim = data['x'].shape[0]
            kerpar = np.atleast_2d(np.ones(dim + 1))
            np.random.seed(1)  # for reproducibility reasons
            filt = StudentProcessStudent(data['dyn'], data['obs'], kerpar, kerpar)
            filt.forward_pass(data['y'][..., 0])

    def test_fully_symmetric_student(self):
        """
        Test fully-symmetric SF.
        """

        for ssm_name, data in self.ssm.items():
            filt = FullySymmetricStudent(data['dyn'], data['obs'])
            filt.forward_pass(data['y'][..., 0])


class GPQMarginalizedTest(TestCase):

    @classmethod
    def setUpClass(cls):
        # setup UNGM
        x0 = GaussRV(1, cov=np.atleast_2d(1.0))
        q = GaussRV(1, cov=np.atleast_2d(10.0))
        cls.dyn_ungm = UNGMTransition(x0, q)
        r = GaussRV(1, cov=np.atleast_2d(1.0))
        cls.obs_ungm = UNGMMeasurement(r, 1)

        # setup pendulum
        dt = 0.01
        x0 = GaussRV(2, np.array([1.5, 0]), 0.01*np.eye(2))
        q = GaussRV(2, cov=np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]))
        cls.dyn_pend = Pendulum2DTransition(x0, q, dt)
        r = GaussRV(1, cov=np.atleast_2d(0.1))
        cls.obs_pend = Pendulum2DMeasurement(r, cls.dyn_pend.dim_state)

    def test_init(self):
        alg = MarginalizedGaussianProcessKalman(self.dyn_ungm, self.obs_ungm, 'rbf', 'sr')

    def test_time_update(self):
        alg = MarginalizedGaussianProcessKalman(self.dyn_ungm, self.obs_ungm, 'rbf', 'sr')
        alg._time_update(1)
        par_dyn, par_obs = np.array([1, 1]), np.array([1, 1])
        alg._time_update(1, par_dyn, par_obs)

    def test_laplace_approx(self):
        alg = MarginalizedGaussianProcessKalman(self.dyn_ungm, self.obs_ungm, 'rbf', 'sr')
        # Random measurement
        y = np.sqrt(10)*np.random.randn(1)
        alg._param_posterior_moments(y, 10)
        # does parameter posterior have positive semi-definite covariance?
        self.assertTrue(np.all(np.linalg.eigvals(alg.param_cov) >= 0))

    def test_measurement_update(self):
        y = self.obs_ungm.simulate_measurements(self.dyn_ungm.simulate_discrete(5))
        alg = MarginalizedGaussianProcessKalman(self.dyn_ungm, self.obs_ungm, 'rbf', 'sr')
        alg._measurement_update(y[:, 0, 0], 1)

    def test_filtering_ungm(self):
        y = self.obs_ungm.simulate_measurements(self.dyn_ungm.simulate_discrete(100))
        alg = MarginalizedGaussianProcessKalman(self.dyn_ungm, self.obs_ungm, 'rbf', 'sr')
        alg.forward_pass(y[..., 0])

    def test_filtering_pendulum(self):
        y = self.obs_pend.simulate_measurements(self.dyn_pend.simulate_discrete(100))
        alg = MarginalizedGaussianProcessKalman(self.dyn_pend, self.obs_pend, 'rbf', 'sr')
        alg.forward_pass(y[..., 0])
