from unittest import TestCase

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from numpy import newaxis as na

from ssmtoybox.bq.bqmod import GaussianProcess, StudentTProcess, BayesSardModel
from ssmtoybox.utils import vandermonde


fcn = lambda x: np.sin((x + 1) ** -1)
# fcn = lambda x: 0.5 * x + 25 * x / (1 + x ** 2)


# fcn = lambda x: np.sin(x)
fcn = lambda x: 0.05*x ** 2
# fcn = lambda x: x


class GPModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ker_par_1d = np.array([[1, 3]])
        cls.ker_par_5d = np.array([[1, 3, 3, 3, 3, 3]])
        cls.pt_par_ut = {'alpha': 1.0}

    def test_init(self):
        GaussianProcess(1, self.ker_par_1d, 'rbf', 'ut', self.pt_par_ut)
        GaussianProcess(5, self.ker_par_5d, 'rbf', 'ut', self.pt_par_ut)

    def test_plotting(self):
        model = GaussianProcess(1, self.ker_par_1d, 'rbf', 'ut', self.pt_par_ut)
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        model.plot_model(xtest, y, fcn_true=f)

    def test_exp_model_variance(self):
        model = GaussianProcess(1, self.ker_par_1d, 'rbf', 'ut', self.pt_par_ut)
        model.bq_weights(self.ker_par_1d)
        self.assertTrue(model.exp_model_variance(self.ker_par_1d) >= 0)

    def test_integral_variance(self):
        model = GaussianProcess(1, self.ker_par_1d, 'rbf', 'ut', self.pt_par_ut)
        self.assertTrue(model.integral_var(self.ker_par_1d) >= 0)

    def test_log_marginal_likelihood(self):
        model = GaussianProcess(1, self.ker_par_1d, 'rbf', 'ut', self.pt_par_ut)
        y = fcn(model.points)
        lhyp = np.log([1.0, 3.0])
        f, df = model.neg_log_marginal_likelihood(lhyp, y.T, model.points, 1e-8*np.eye(model.num_pts))

    @staticmethod
    def _nlml(log_par, kernel, fcn_obs, x_obs):

        # convert from log-par to par
        par = np.exp(log_par)
        num_data = x_obs.shape[1]

        K = kernel.eval(par, x_obs)  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, )
        y_dot_a = fcn_obs.dot(a)

        return np.sum(np.log(np.diag(L[0]))) + 0.5 * (y_dot_a + num_data * np.log(2 * np.pi))

    @staticmethod
    def _nlml_grad(log_par, kernel, fcn_obs, x_obs):
        # convert from log-par to par
        par = np.exp(log_par)

        num_data = x_obs.shape[1]
        K = kernel.eval(par, x_obs)  # (N, N)
        L = la.cho_factor(K)
        a = la.cho_solve(L, fcn_obs)  # (N, )
        a_out_a = np.outer(a, a.T)  # (N, N) sum over of outer products of columns of A

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = kernel.der_par(par, x_obs)  # (N, N, num_par)
        # iK = la.solve(K, np.eye(num_data))
        # return 0.5 * np.trace((iK - a_out_a).dot(dK_dTheta))  # (num_par, )
        iKdK = la.cho_solve(L, dK_dTheta)
        return 0.5 * np.trace((iKdK - a_out_a.dot(dK_dTheta)))  # (num_par, )

    def test_nlml_gradient(self):
        model = GaussianProcess(5, self.ker_par_5d, 'rbf', 'ut', self.pt_par_ut)
        y = fcn(model.points)
        lhyp = np.log([1.0] + 5*[3.0])

        from scipy.optimize import check_grad
        err = check_grad(self._nlml, self._nlml_grad, lhyp, model.kernel, y.T[:, 0], model.points)
        print(err)
        self.assertTrue(err <= 1e-5, 'Gradient error: {:.4f}'.format(err))

    @staticmethod
    def _total_nlml(log_par, kernel, fcn_obs, x_obs):
        # N - # points, E - # function outputs
        # fcn_obs (N, E), hypers (num_hyp, )

        # convert from log-par to par
        par = np.exp(log_par)
        num_data = x_obs.shape[1]
        num_out = fcn_obs.shape[1]

        K = kernel.eval(par, x_obs)  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ji', fcn_obs.T, a)  # sum of diagonal of A.T.dot(A)

        # negative marginal log-likelihood
        return num_out * np.sum(np.log(np.diag(L[0]))) + 0.5 * (y_dot_a + num_out*num_data*np.log(2 * np.pi))

    @staticmethod
    def _total_nlml_grad(log_par, kernel, fcn_obs, x_obs):
        # N - # points, E - # function outputs
        # fcn_obs (N, E), hypers (num_hyp, )

        # convert from log-par to par
        par = np.exp(log_par)
        num_data = x_obs.shape[1]
        num_out = fcn_obs.shape[1]

        K = kernel.eval(par, x_obs)  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, E)
        a_out_a = np.einsum('i...j, ...jn', a, a.T)  # (N, N) sum over of outer products of columns of A

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = kernel.der_par(par, x_obs)  # (N, N, num_hyp)
        iKdK = la.cho_solve(L, dK_dTheta)

        # gradient of total NLML
        return 0.5 * np.trace((num_out * iKdK - a_out_a.dot(dK_dTheta)))  # (num_par, )

    def test_total_nlml_gradient(self):

        # nonlinear vector function from some SSM
        from ssmtoybox.ssmod import CoordinatedTurnRadar
        ssm = CoordinatedTurnRadar()

        # generate inputs
        num_x = 20
        x = 10 + np.random.randn(ssm.xD, num_x)

        # evaluate function at inputs
        y = np.apply_along_axis(ssm.dyn_eval, 0, x, None)

        # kernel and it's initial parameters
        from ssmtoybox.bq.bqkern import RBF
        lhyp = np.log([1.0] + 5 * [3.0])
        kernel = RBF(ssm.xD, self.ker_par_5d)

        from scipy.optimize import check_grad
        err = check_grad(self._total_nlml, self._total_nlml_grad, lhyp, kernel, y.T, x)
        print(err)
        self.assertTrue(err <= 1e-5, 'Gradient error: {:.4f}'.format(err))

    def test_hypers_optim(self):
        model = GaussianProcess(1, self.ker_par_1d, 'rbf', 'gh', point_par={'degree': 15})
        xtest = np.linspace(-7, 7, 100)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        # plot before optimization
        # model.plot_model(xtest, y, fcn_true=f)
        lhyp0 = np.log([[1.0, 0.5]])
        b = ((np.log(0.9), np.log(1.1)), (None, None))

        def con_alpha(lhyp):
            # constrain alpha**2 = 1
            return np.exp(lhyp[0]) ** 2 - 1 ** 2

        con = {'type': 'eq', 'fun': con_alpha}
        res_ml2 = model.optimize(lhyp0, y.T, model.points, method='BFGS', constraints=con)
        hyp_ml2 = np.exp(res_ml2.x)

        print(res_ml2)

        print('ML-II({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1]))

        # plot after optimization
        model.plot_model(xtest, y, fcn_true=f, par=hyp_ml2)

        # TODO: test fitting of multioutput GPs, GPy supports this in GPRegression
        # plot NLML surface
        # x = np.log(np.mgrid[1:10:0.5, 0.5:20:0.5])
        # m, n = x.shape[1:]
        # z = np.zeros(x.shape[1:])
        # for i in range(m):
        #     for j in range(n):
        #         z[i, j], grad = model.neg_log_marginal_likelihood(x[:, i, j], y.T)
        #
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface((x[0, ...]), (x[1, ...]), z, linewidth=0.5, alpha=0.5, rstride=2, cstride=2)
        # ax.set_xlabel('alpha')
        # ax.set_ylabel('el')
        # plt.show()

    def test_hypers_optim_multioutput(self):
        from ssmtoybox.ssmod import CoordinatedTurnRadar
        ssm = CoordinatedTurnRadar()
        func = ssm.dyn_eval
        dim_in, dim_out = ssm.xD, ssm.xD

        model = GaussianProcess(dim_in, self.ker_par_5d, 'rbf', 'sr')  # , point_hyp={'degree': 10})
        x = ssm.get_pars('x0_mean')[0][:, na] + model.points  # ssm.get_pars('x0_cov')[0].dot(model.points)
        y = np.apply_along_axis(func, 0, x, None)  # (d_out, n**2)

        # lhyp0 = np.log(np.ones((dim_out, dim_in+1)))
        lhyp0 = np.log(np.ones((1, dim_in + 1)))

        res_ml2 = model.optimize(lhyp0, y.T, model.points, method='BFGS')
        hyp_ml2 = np.exp(res_ml2.x)

        print(res_ml2)
        np.set_printoptions(precision=4)
        print('ML-II({:.4f}) @ alpha: {:.4f}, el: {}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1:]))


class BayesSardModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ker_par_1d = np.array([[1.0, 1]])
        cls.ker_par_2d = np.array([[1.0, 1.0, 1.0]])
        cls.ker_par_5d = np.array([[1.0, 3, 3, 3, 3, 3]])
        cls.data_1d = np.array([[1, -1, 0]], dtype=float)
        cls.data_2d = np.hstack((np.zeros((2, 1)), np.eye(2), -np.eye(2)))
        cls.pt_par_ut = {'alpha': 1.0}

    def test_init(self):
        BayesSardModel(1, self.ker_par_1d, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)

    def test_prediction(self):
        model = BayesSardModel(1, self.ker_par_1d, multi_ind=2, point_str='gh', point_par={'degree': 5})
        xtest = np.linspace(-5, 5, 100)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        alpha = np.array([[0, 1, 2]])
        mean, var = model.predict(xtest, y, mulind=alpha)
        std = np.sqrt(var)

        # plot training data, predictive mean and variance
        fig_title = 'BSQ model predictions'
        fig = plt.figure(fig_title)
        xtest = np.squeeze(xtest)
        plt.fill_between(xtest, mean - 2 * std, mean + 2 * std, color='0.1', alpha=0.15)
        plt.plot(xtest, mean, color='k', lw=2)
        plt.plot(model.points, y, 'ko', ms=8)

        # true function values at test points if provided
        if f is not None:
            plt.plot(xtest, np.squeeze(f), lw=2, ls='--', color='tomato')
        plt.show()

    def test_x_px(self):
        model = BayesSardModel(1, self.ker_par_1d, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)
        mi_1d = np.array([[0, 1, 2]])
        ke = model._exp_x_px(mi_1d)
        self.assertTrue(ke.shape == (mi_1d.shape[1], ))
        self.assertTrue(np.array_equal(ke, np.array([1, 0, 1])))

        model = BayesSardModel(2, self.ker_par_2d, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)
        mi_2d = np.array([[0, 1, 0, 1, 0, 2],
                          [0, 0, 1, 1, 2, 0]])
        ke = model._exp_x_px(mi_2d)
        ke_true = np.array([1, 0, 0, 0, 1, 1])
        self.assertTrue(ke.shape == (mi_2d.shape[1], ))
        self.assertTrue(np.array_equal(ke, ke_true))

    def test_exp_x_xpx(self):
        model = BayesSardModel(1, self.ker_par_1d, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)
        mi_1d = np.array([[0, 1, 2]])
        ke = model._exp_x_xpx(mi_1d)
        self.assertTrue(ke.shape == mi_1d.shape)
        self.assertTrue(np.array_equal(ke, np.array([[0, 1, 0]])))

        model = BayesSardModel(2, self.ker_par_2d, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)
        mi_2d = np.array([[0, 1, 0, 1, 0, 2],
                          [0, 0, 1, 1, 2, 0]])
        ke_true = np.array([[0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0]])
        ke = model._exp_x_xpx(mi_2d)
        self.assertTrue(ke.shape == mi_2d.shape)
        self.assertTrue(np.array_equal(ke, ke_true))

    def test_exp_x_pxpx(self):
        model = BayesSardModel(1, self.ker_par_1d, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)
        mi_1d = np.array([[0, 1, 2]])
        ke = model._exp_x_pxpx(mi_1d)
        ke_true = np.array([[1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 3]])
        self.assertTrue(ke.shape == (mi_1d.shape[1], mi_1d.shape[1]))
        self.assertTrue(np.array_equal(ke, ke_true))

        model = BayesSardModel(2, self.ker_par_2d, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)
        mi_2d = np.array([[0, 1, 0, 1, 0, 2],
                          [0, 0, 1, 1, 2, 0]])
        ke_true = np.array([[1, 0, 0, 0, 1, 1],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [1, 0, 0, 0, 3, 1],
                            [1, 0, 0, 0, 1, 3]])
        ke = model._exp_x_pxpx(mi_2d)
        self.assertTrue(ke.shape == (mi_2d.shape[1], mi_2d.shape[1]))
        self.assertTrue(np.array_equal(ke, ke_true))

    def test_exp_x_kxpx(self):
        model = BayesSardModel(1, self.ker_par_1d, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)
        mi_1d = np.array([[0, 1, 2]])
        par_1d = np.array([[1.0, 1.0]])
        data = np.array([[0, 1, -1]], dtype=np.float)
        ke = model._exp_x_kxpx(par_1d, mi_1d, data)
        ke_true = np.array([[2**(-0.5), 0, 1/(2*(2**0.5))],
                            [2**(-0.5)*np.exp(-0.25), np.exp(-0.25)/(2*(2**0.5)), 3*np.exp(-0.25)/(4*2**0.5)],
                            [2**(-0.5)*np.exp(-0.25), -np.exp(-0.25)/(2*(2**0.5)), 3*np.exp(-0.25)/(4*2**0.5)]])
        self.assertTrue(ke.shape == (data.shape[1], mi_1d.shape[1]))
        self.assertTrue(np.allclose(ke, ke_true))

    def test_mc_poly_verification(self):
        dim = 1
        alpha = np.array([[0, 1, 2]])
        par = np.array([[1.0, 1]])
        model = BayesSardModel(1, par, multi_ind=2, point_str='ut', point_par=self.pt_par_ut)
        px = model._exp_x_px(alpha)
        xpx = model._exp_x_xpx(alpha)
        pxpx = model._exp_x_pxpx(alpha)
        kxpx = model._exp_x_kxpx(par, alpha, self.data_1d)

        # approximate expectations using cumulative moving average MC
        def cma_mc(new_samples, old_avg, old_avg_size, axis=0):
            b_size = new_samples.shape[axis]
            return (new_samples.sum(axis=axis) + old_avg_size * old_avg) / (old_avg_size + b_size)

        batch_size = 100000
        num_iter = 100
        px_mc, xpx_mc, pxpx_mc, kxpx_mc = 0, 0, 0, 0
        for i in range(num_iter):
            # sample from standard Gaussian
            x_samples = np.random.multivariate_normal(np.zeros((dim, )), np.eye(dim), size=batch_size).T
            p = vandermonde(alpha, x_samples)  # (N, Q)
            k = model.kernel.eval(par, x_samples, self.data_1d, scaling=False)  # (N, M)
            px_mc = cma_mc(p, px_mc, i*batch_size, axis=0)
            xpx_mc = cma_mc(x_samples[..., na] * p[na, ...], xpx_mc, i*batch_size, axis=1)
            pxpx_mc = cma_mc(p[:, na, :] * p[..., na], pxpx_mc, i*batch_size, axis=0)
            kxpx_mc = cma_mc(k[..., na] * p[:, na, :], kxpx_mc, i*batch_size, axis=0)

        # compare MC approximates with analytic expressions
        tol = 1e-3
        print('Maximum absolute difference using {:d} samples.'.format(batch_size*num_iter))
        print('px {:.2e}'.format(np.abs(px - px_mc).max()))
        print('xpx {:.2e}'.format(np.abs(xpx - xpx_mc).max()))
        print('pxpx {:.2e}'.format(np.abs(pxpx - pxpx_mc).max()))
        print('kxpx {:.2e}'.format(np.abs(kxpx - kxpx_mc).max()))
        self.assertLessEqual(np.abs(px - px_mc).max(), tol)
        self.assertLessEqual(np.abs(xpx - xpx_mc).max(), tol)
        self.assertLessEqual(np.abs(pxpx - pxpx_mc).max(), tol)
        self.assertLessEqual(np.abs(kxpx - kxpx_mc).max(), tol)

    def test_weights(self):
        from ssmtoybox.mtran import Unscented, GaussHermite

        # UT weights in 1D
        model = BayesSardModel(1, self.ker_par_1d, point_str='ut', point_par=self.pt_par_ut)
        alpha = np.array([[0, 1, 2]])
        w, wc, wcc, emv, ivar = model.bq_weights(self.ker_par_1d, alpha)
        # UT weights in 1D reproduced?
        self.assertTrue(np.allclose(w, Unscented.weights(1)[0]))
        self.assertGreaterEqual(emv, 0)
        self.assertGreaterEqual(ivar, 0)
        # test positive definiteness
        try:
            la.cholesky(wc)
        except la.LinAlgError:
            self.fail("Weights not positive definite. Min eigval: {}".format(la.eigvalsh(wc).min()))

        # UT weights in 2D
        par = np.array([[1.0, 1.0, 1]])
        alpha = np.array([[0, 1, 0, 2, 0],
                          [0, 0, 1, 0, 2]])
        model = BayesSardModel(2, par, point_str='ut', point_par=self.pt_par_ut)
        w, wc, wcc, emv, ivar = model.bq_weights(par, alpha)
        # UT weights reproduced in 2D?
        self.assertTrue(np.allclose(w, Unscented.weights(2)[0]))
        self.assertGreaterEqual(emv, 0)
        self.assertGreaterEqual(ivar, 0)
        # test positive definiteness
        try:
            la.cholesky(wc)
        except la.LinAlgError:
            self.fail("Weights not positive definite. Min eigval: {}".format(la.eigvalsh(wc).min()))

        # GH-3 weights in 2D
        # there are 6 multivariate polynomials in 2D, UT has only 5 points in 2D
        model = BayesSardModel(2, self.ker_par_2d, point_str='gh', point_par={'degree': 3})
        alpha = np.array([[0, 1, 0, 1, 2, 0, 1, 2, 2],
                          [0, 0, 1, 1, 0, 2, 2, 1, 2]])
        par = np.array([[1.0, 1, 1]])
        w, wc, wcc, emv, ivar = model.bq_weights(par, alpha)
        self.assertTrue(np.allclose(w, GaussHermite.weights(2, 3)))
        self.assertGreaterEqual(emv, 0)
        self.assertGreaterEqual(ivar, 0)
        # test positive definiteness
        try:
            la.cholesky(wc)
        except la.LinAlgError:
            self.fail("Weights not positive definite. Min eigval: {}".format(la.eigvalsh(wc).min()))


class TPModelTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ker_par_1d = np.array([[1, 3]])
        cls.ker_par_5d = np.array([[1, 3, 3, 3, 3, 3]])
        cls.pt_par_ut = {'alpha': 1.0}

    def test_init(self):
        StudentTProcess(1, self.ker_par_1d)
        StudentTProcess(5, self.ker_par_5d, point_par=self.pt_par_ut)

    def test_plotting(self):
        model = StudentTProcess(1, self.ker_par_1d)
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        model.plot_model(xtest, y, fcn_true=f)

    def test_exp_model_variance(self):
        model = StudentTProcess(1, self.ker_par_1d)
        model.bq_weights(self.ker_par_1d)
        y = fcn(model.points)
        self.assertTrue(model.exp_model_variance(self.ker_par_1d, y) >= 0)

    def test_integral_variance(self):
        model = StudentTProcess(1, self.ker_par_1d)
        y = fcn(model.points)
        self.assertTrue(model.integral_variance(self.ker_par_1d, y) >= 0)

    @staticmethod
    def _nlml(log_par, kernel, fcn_obs, x_obs, jitter, nu):
        # convert from log-par to par
        par = np.exp(log_par)
        num_data, num_out = fcn_obs.shape

        K = kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ij -> j', fcn_obs, a)  # sum of diagonal of A.T.dot(A)

        # negative marginal log-likelihood
        from scipy.special import gamma
        half_logdet_K = np.sum(np.log(np.diag(L[0])))
        const = (num_data/2)*np.log((nu-2)*np.pi) - np.log(gamma((nu+num_data)/2)) + np.log(gamma(nu/2))
        log_sum = 0.5*(nu+num_data) * np.log(1 + y_dot_a / (nu - 2)).sum()

        return log_sum + num_out * (half_logdet_K + const)

    @staticmethod
    def _nlml_grad(log_par, kernel, fcn_obs, x_obs, jitter, nu):
        # convert from log-par to par
        par = np.exp(log_par)
        num_data, num_out = fcn_obs.shape

        K = kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ij -> j', fcn_obs, a)  # sum of diagonal of A.T.dot(A)

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = kernel.der_par(par, x_obs)  # (N, N, num_par)

        # gradient
        iKdK = la.cho_solve(L, dK_dTheta)
        scale = (nu + num_data) / (nu + y_dot_a - 2)
        a_out_a = np.einsum('j, i...j, ...jn', scale, a, a.T)  # (N, N) weighted sum of outer products of columns of A

        return 0.5 * np.trace((num_out * iKdK - a_out_a.dot(dK_dTheta)))  # (num_par, )

    def test_nlml_gradient(self):
        model = StudentTProcess(5, self.ker_par_5d, 'rbf', 'ut', self.pt_par_ut)
        y = fcn(model.points)
        lhyp = np.log([1.0] + 5 * [3.0])
        jitter = 1e-8 * np.eye(model.num_pts)
        dof = 3

        from scipy.optimize import check_grad
        err = check_grad(self._nlml, self._nlml_grad, lhyp, model.kernel, y.T, model.points, jitter, dof)
        print(err)
        self.assertTrue(err <= 1e-5, 'Gradient error: {:.4f}'.format(err))

    def test_hypers_optim(self):
        model = StudentTProcess(1, self.ker_par_1d, 'rbf', 'gh', point_par={'degree': 10})
        xtest = np.linspace(-7, 7, 100)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        # plot before optimization
        # model.plot_model(xtest, y, fcn_true=f)
        lhyp0 = np.log([[1.0, 0.5]])
        b = ((np.log(0.9), np.log(1.1)), (None, None))

        def con_alpha(lhyp):
            # constrain alpha**2 = 1
            return np.exp(lhyp[0]) ** 2 - 1 ** 2

        con = {'type': 'eq', 'fun': con_alpha}
        res_ml2 = model.optimize(lhyp0, y.T, model.points, method='BFGS', constraints=con)
        hyp_ml2 = np.exp(res_ml2.x)

        print(res_ml2)

        print('ML-II({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1]))

        # plot after optimization
        model.plot_model(xtest, y, fcn_true=f, par=hyp_ml2)

    def test_hypers_optim_multioutput(self):
        from ssmtoybox.ssmod import CoordinatedTurnRadar
        ssm = CoordinatedTurnRadar()
        func = ssm.dyn_eval
        dim_in, dim_out = ssm.xD, ssm.xD

        model = StudentTProcess(dim_in, self.ker_par_5d, 'rbf', 'ut', nu=50.0)  # , point_hyp={'degree': 10})
        x = ssm.get_pars('x0_mean')[0][:, na] + model.points  # ssm.get_pars('x0_cov')[0].dot(model.points)
        y = np.apply_along_axis(func, 0, x, None)  # (d_out, n**2)

        # lhyp0 = np.log(np.ones((dim_out, dim_in+1)))
        lhyp0 = np.log(10 + np.ones((1, dim_in + 1)))

        res_ml2 = model.optimize(lhyp0, y.T, model.points, method='BFGS')
        hyp_ml2 = np.exp(res_ml2.x)

        print(res_ml2)
        np.set_printoptions(precision=4)
        print('ML-II({:.4f}) @ alpha: {:.4f}, el: {}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1:]))