import unittest
from unittest import TestCase

import numpy as np
import numpy.linalg as la

from ssmtoybox.bq.bqmtran import GaussianProcessTransform, MultiOutputGaussianProcessTransform, BayesSardTransform
from ssmtoybox.ssmod import UNGMTransition, Pendulum2DTransition, CoordinatedTurnTransition, ReentryVehicle2DTransition
from ssmtoybox.utils import GaussRV

np.set_printoptions(precision=4)


class GPQuadTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = []
        cls.models.append(UNGMTransition(GaussRV(1), GaussRV(1)))
        cls.models.append(Pendulum2DTransition(GaussRV(2), GaussRV(2)))

    def test_weights_rbf(self):
        dim = 1
        khyp = np.array([[1, 3]], dtype=np.float)
        phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = GaussianProcessTransform(dim, 1, khyp, point_par=phyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        print('wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc))
        self.assertTrue(np.allclose(wc, wc.T), "Covariance weight matrix not symmetric.")
        # print 'GP model variance: {}'.format(tf.model.exp_model_variance())

        dim = 2
        khyp = np.array([[1, 3, 3]], dtype=np.float)
        phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = GaussianProcessTransform(dim, 1, khyp, point_par=phyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        print('wm = \n{}\nwc = \n{}\nwcc = \n{}'.format(wm, wc, wcc))
        self.assertTrue(np.allclose(wc, wc.T), "Covariance weight matrix not symmetric.")

    def test_rbf_scaling_invariance(self):
        dim = 5
        ker_par = np.array([[1, 3, 3, 3, 3, 3]], dtype=np.float)
        tf = GaussianProcessTransform(dim, 1, ker_par)
        w0 = tf.weights([1] + dim * [1000])
        w1 = tf.weights([358.0] + dim * [1000.0])
        self.assertTrue(np.alltrue([np.array_equal(a, b) for a, b in zip(w0, w1)]))

    def test_expected_model_variance(self):
        dim = 2
        ker_par = np.array([[1, 3, 3]], dtype=np.float)
        tf = GaussianProcessTransform(dim, 1, ker_par, point_str='sr')
        emv0 = tf.model.exp_model_variance(ker_par)
        emv1 = tf.model.exp_model_variance(ker_par)
        # expected model variance must be positive even for numerically unpleasant settings
        self.assertTrue(np.alltrue(np.array([emv0, emv1]) >= 0))

    def test_integral_variance(self):
        dim = 2
        ker_par = np.array([[1, 3, 3]], dtype=np.float)
        tf = GaussianProcessTransform(dim, 1, ker_par, point_str='sr')
        ivar0 = tf.model.integral_variance([1, 600, 6])
        ivar1 = tf.model.integral_variance([1.1, 600, 6])
        # expected model variance must be positive even for numerically unpleasant settings
        self.assertTrue(np.alltrue(np.array([ivar0, ivar1]) >= 0))

    def test_apply(self):
        for mod in self.models:
            f = mod.dyn_eval
            dim = mod.dim_in
            ker_par = np.hstack((np.ones((1, 1)), 3*np.ones((1, dim))))
            tf = GaussianProcessTransform(dim, dim, ker_par)
            mean, cov = np.zeros(dim, ), np.eye(dim)
            tmean, tcov, tccov = tf.apply(f, mean, cov, np.atleast_1d(1.0))
            print("Transformed moments\nmean: {}\ncov: {}\nccov: {}".format(tmean, tcov, tccov))

            self.assertTrue(tf.I_out.shape == (dim, dim))
            # test positive definiteness
            try:
                la.cholesky(tcov)
            except la.LinAlgError:
                self.fail("Output covariance not positive definite.")

            # test symmetry
            self.assertTrue(np.allclose(tcov, tcov.T), "Output covariance not closely symmetric.")
            # self.assertTrue(np.array_equal(tcov, tcov.T), "Output covariance not exactly symmetric.")


class BSQTransformTest(TestCase):
    def test_polar2cartesian(self):
        def polar2cartesian(x, pars):
            return x[0] * np.array([np.cos(x[1]), np.sin(x[1])])

        mean_in = np.array([1, np.pi / 2])
        cov_in = np.diag([0.05 ** 2, (np.pi / 10) ** 2])
        alpha_ut = np.array([[0, 1, 0, 2, 0],
                             [0, 0, 1, 0, 2]])
        par = np.array([[1.0, 1, 1]])
        mt = BayesSardTransform(2, 2, par, multi_ind=alpha_ut, point_str='ut')
        mean_out, cov_out, cc = mt.apply(polar2cartesian, mean_in, cov_in, np.atleast_1d(0))
        self.assertTrue(mt.I_out.shape == (2, 2))
        try:
            la.cholesky(cov_out)
        except la.LinAlgError:
            self.fail("Weights not positive definite. Min eigval: {}".format(la.eigvalsh(cov_out).min()))


@unittest.skip('Multi-output models are experimental and not a priority.')
class GPQMOTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = []
        cls.models.append(UNGMTransition(GaussRV(1), GaussRV(1)))
        cls.models.append(Pendulum2DTransition(GaussRV(2), GaussRV(2)))

    def test_weights_rbf(self):
        dim_in, dim_out = 1, 1
        khyp = np.array([[1, 3]])
        phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = MultiOutputGaussianProcessTransform(dim_in, dim_out, khyp, point_par=phyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        self.assertTrue(np.allclose(wc, wc.swapaxes(0, 1).swapaxes(2, 3)), "Covariance weight matrix not symmetric.")

        dim_in, dim_out = 4, 4
        khyp = np.array([[1, 3, 3, 3, 3],
                         [1, 1, 1, 1, 1],
                         [1, 2, 2, 2, 2],
                         [1, 3, 3, 3, 3]])
        phyp = {'kappa': 0.0, 'alpha': 1.0}
        tf = MultiOutputGaussianProcessTransform(dim_in, dim_out, khyp, point_par=phyp)
        wm, wc, wcc = tf.wm, tf.Wc, tf.Wcc
        self.assertTrue(np.allclose(wc, wc.swapaxes(0, 1).swapaxes(2, 3)), "Covariance weight matrix not symmetric.")

    def test_apply(self):
        dyn = Pendulum2DTransition(GaussRV(2), GaussRV(2))
        f = dyn.dyn_eval
        dim_in, dim_out = dyn.dim_in, dyn.dim_state
        ker_par = np.hstack((np.ones((dim_out, 1)), 3*np.ones((dim_out, dim_in))))
        tf = MultiOutputGaussianProcessTransform(dim_in, dim_out, ker_par)
        mean, cov = np.zeros(dim_in, ), np.eye(dim_in)
        tmean, tcov, tccov = tf.apply(f, mean, cov, np.atleast_1d(1.0))
        print("Transformed moments\nmean: {}\ncov: {}\nccov: {}".format(tmean, tcov, tccov))

        # test positive definiteness
        try:
            la.cholesky(tcov)
        except la.LinAlgError:
            self.fail("Output covariance not positive definite.")

        # test symmetry
        self.assertTrue(np.allclose(tcov, tcov.T), "Output covariance not closely symmetric.")
        # self.assertTrue(np.array_equal(tcov, tcov.T), "Output covariance not exactly symmetric.")

    def test_single_vs_multi_output(self):
        # results of the GPQ and GPQMO should be same if parameters properly chosen, GPQ is a special case of GPQMO
        m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932])
        P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
        x0 = GaussRV(5, m0, P0)
        dyn = ReentryVehicle2DTransition(x0, GaussRV(5))
        f = dyn.dyn_eval
        dim_in, dim_out = dyn.dim_in, dyn.dim_state

        # input mean and covariance
        mean_in, cov_in = m0, P0

        # single-output GPQ
        ker_par_so = np.hstack((np.ones((1, 1)), 25 * np.ones((1, dim_in))))
        tf_so = GaussianProcessTransform(dim_in, dim_out, ker_par_so)

        # multi-output GPQ
        ker_par_mo = np.hstack((np.ones((dim_out, 1)), 25 * np.ones((dim_out, dim_in))))
        tf_mo = MultiOutputGaussianProcessTransform(dim_in, dim_out, ker_par_mo)

        # transformed moments
        # FIXME: transformed covariances different
        mean_so, cov_so, ccov_so = tf_so.apply(f, mean_in, cov_in, np.atleast_1d(0))
        mean_mo, cov_mo, ccov_mo = tf_mo.apply(f, mean_in, cov_in, np.atleast_1d(0))

        print('mean delta: {}'.format(np.abs(mean_so - mean_mo).max()))
        print('cov delta: {}'.format(np.abs(cov_so - cov_mo).max()))
        print('ccov delta: {}'.format(np.abs(ccov_so - ccov_mo).max()))

        # results of GPQ and GPQMO should be the same
        self.assertTrue(np.array_equal(mean_so, mean_mo))
        self.assertTrue(np.array_equal(cov_so, cov_mo))
        self.assertTrue(np.array_equal(ccov_so, ccov_mo))

    def test_optimize_1D(self):
        # test on simple 1D example, plot the fit
        steps = 100
        dyn = UNGMTransition(GaussRV(1), GaussRV(1))
        x, y = dyn.simulate_discrete(steps)

        f = dyn.dyn_eval
        dim_in, dim_out = dyn.dim_in, dyn.dim_state

        par0 = 1 + np.random.rand(dim_out, dim_in + 1)
        tf = MultiOutputGaussianProcessTransform(dim_in, dim_out, par0)

        # use sampled system state trajectory to create training data
        fy = np.zeros((dim_out, steps))
        for k in range(steps):
            fy[:, k] = f(x[:, k, 0], k)

        b = [np.log((0.1, 1.0001))] + dim_in * [(None, None)]
        opt = {'xtol': 1e-2, 'maxiter': 100}
        log_par, res_list = tf.model.optimize(np.log(par0), fy, x[..., 0], bounds=b, method='L-BFGS-B', options=opt)

        print(np.exp(log_par))
        self.assertTrue(False)

    def test_optimize(self):
        steps = 350
        m0 = np.array([1000, 300, 1000, 0, np.deg2rad(-3)])
        P0 = np.diag([100, 10, 100, 10, 0.1])
        x0 = GaussRV(5, m0, P0)
        dt = 0.1
        Q = np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        q = GaussRV(5, cov=Q)
        dyn = CoordinatedTurnTransition(x0, q, dt)
        x, y = dyn.simulate_discrete(steps)

        f = dyn.dyn_eval
        dim_in, dim_out = dyn.dim_in, dyn.dim_state

        # par0 = np.hstack((np.ones((dim_out, 1)), 5*np.ones((dim_out, dim_in+1))))
        par0 = 10*np.ones((dim_out, dim_in+1))
        tf = MultiOutputGaussianProcessTransform(dim_in, dim_out, par0)

        # use sampled system state trajectory to create training data
        fy = np.zeros((dim_out, steps))
        for k in range(steps):
            fy[:, k] = f(x[:, k, 0], 0)

        opt = {'maxiter': 100}
        log_par, res_list = tf.model.optimize(np.log(par0), fy, x[..., 0], method='BFGS', options=opt)

        print(np.exp(log_par))