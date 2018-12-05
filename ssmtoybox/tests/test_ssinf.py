from unittest import TestCase

import numpy as np
import numpy.linalg as la

from ssmtoybox.ssinf import GPQMKalman, GaussianProcessKalman, BayesSardKalman, TPQStudent, TPQKalman
from ssmtoybox.ssinf import UnscentedKalman, ExtendedKalman, GaussHermiteKalman
from ssmtoybox.ssmod import UNGMTransition, UNGMNATransition, Pendulum2DTransition, CoordinatedTurnTransition, \
    ReentryVehicle2DTransition, ConstantTurnRateSpeed, ConstantVelocity
from ssmtoybox.ssmod import UNGMMeasurement, UNGMNAMeasurement, Pendulum2DMeasurement, BearingMeasurement, \
    Radar2DMeasurement
from ssmtoybox.utils import GaussRV, StudentRV


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
                alg = TPQKalman(data['dyn'], data['obs'], kpar_dyn, kpar_obs)
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
        m_0 = np.array([10000, 300, 1000, -40]).astype(np.float)
        P_0 = np.diag([10000, 100, 10000, 100]).astype(np.float)
        nu_0 = 1000.0
        x0 = StudentRV(4, m_0, P_0, nu_0)
        Q = np.diag([50, 5]).astype(np.float)
        nu_q = 1000.0
        q = StudentRV(2, scale=Q, dof=nu_q)
        R = np.diag([50, 0.4e-6]).astype(np.float)
        nu_r = 4.0
        r = StudentRV(2, scale=R, dof=nu_r)
        dyn = ConstantVelocity(x0, q)
        obs = Radar2DMeasurement(r, 4)
        x = dyn.simulate_discrete(100)
        y = obs.simulate_measurements(x)
        cls.ssm.update({'cv': {'dyn': dyn, 'obs': obs, 'x': x, 'y': y}})

    @staticmethod
    def rbf_student_mc_weights(x, kern, num_samples, num_batch):
        # MC approximated BQ weights using RBF kernel and Student density
        # MC computed by batches, because without batches we would run out of memory for large sample sizes
        from ssmtoybox.bq.bqkern import RBFStudent
        from ssmtoybox.utils import multivariate_t
        from numpy import newaxis as na

        assert isinstance(kern, RBFStudent)
        # kernel parameters and input dimensionality
        par = kern.par
        dim, num_pts = x.shape

        # inverse kernel matrix
        iK = kern.eval_inv_dot(kern.par, x, scaling=False)
        mean, scale, dof = np.zeros((dim,)), np.eye(dim), kern.dof

        # compute MC estimates by batches
        num_samples_batch = num_samples // num_batch
        q_batch = np.zeros((num_pts, num_batch,))
        Q_batch = np.zeros((num_pts, num_pts, num_batch))
        R_batch = np.zeros((dim, num_pts, num_batch))
        for ib in range(num_batch):
            # multivariate t samples
            x_samples = multivariate_t(mean, scale, dof, num_samples_batch).T

            # evaluate kernel
            k_samples = kern.eval(par, x_samples, x, scaling=False)
            kk_samples = k_samples[:, na, :] * k_samples[..., na]
            xk_samples = x_samples[..., na] * k_samples[na, ...]

            # intermediate sums
            q_batch[..., ib] = k_samples.sum(axis=0)
            Q_batch[..., ib] = kk_samples.sum(axis=0)
            R_batch[..., ib] = xk_samples.sum(axis=1)

        # MC approximations == sum the sums divide by num_samples
        c = 1 / num_samples
        q = c * q_batch.sum(axis=-1)
        Q = c * Q_batch.sum(axis=-1)
        R = c * R_batch.sum(axis=-1)

        # BQ moment transform weights
        wm = q.dot(iK)
        wc = iK.dot(Q).dot(iK)
        wcc = R.dot(iK)
        return wm, wc, wcc, Q

    def test_student_process_student(self):
        """
        Test t-Process Quadrature SF on a range of SSMs.
        """
        # for ssm_name, data in self.ssm.items():
        #     dim = data['x'].shape[0]
        #     kerpar = np.atleast_2d(np.ones(dim + 1))
        #     filt = TPQStudent(data['dyn'], data['obs'], kerpar, kerpar)
        #     filt.forward_pass(data['y'][..., 0])

        dyn = self.ssm['cv']['dyn']
        obs = self.ssm['cv']['obs']
        kpar_dyn = np.array([[1.0, 100, 100, 100, 100]])
        kpar_obs = np.array([[0.05, 10, 100, 10, 100]])
        filt = TPQStudent(dyn, obs, kpar_dyn, kpar_obs, point_par={'kappa': 0.0})

        # the weights of the TPQSF with RBF kernel need large number of MC samples (2M) to determine confidently
        np.random.seed(4)
        wm, wc, wcc, Q = self.rbf_student_mc_weights(filt.tf_dyn.model.points, filt.tf_dyn.model.kernel, int(2e6), 1000)
        filt.tf_dyn.wm, filt.tf_dyn.Wc, filt.tf_dyn.Wcc = wm, wc, wcc
        np.random.seed(4)
        wm, wc, wcc, Q = self.rbf_student_mc_weights(filt.tf_meas.model.points, filt.tf_meas.model.kernel, int(2e6), 1000)
        filt.tf_meas.wm, filt.tf_meas.Wc, filt.tf_meas.Wcc = wm, wc, wcc

        filt.forward_pass(self.ssm['cv']['y'][..., 0])

    def test_fully_symmetric_student(self):
        """
        Test fully-symmetric SF.
        """
        self.fail("Not implemented yet!")


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