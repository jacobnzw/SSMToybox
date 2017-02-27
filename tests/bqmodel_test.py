from unittest import TestCase

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from numpy import newaxis as na
from transforms.bqmodel import GaussianProcess, StudentTProcess, GaussianProcessMO

# fcn = lambda x: np.sin((x + 1) ** -1)
fcn = lambda x: 0.5 * x + 25 * x / (1 + x ** 2)


# fcn = lambda x: np.sin(x)
# fcn = lambda x: 0.05*x ** 2
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
        y = fcn(model.points)
        self.assertTrue(model.exp_model_variance(y) >= 0)

    def test_integral_variance(self):
        model = GaussianProcess(1, self.ker_par_1d, 'rbf', 'ut', self.pt_par_ut)
        y = fcn(model.points)
        self.assertTrue(model.integral_variance(y) >= 0)

    def test_log_marginal_likelihood(self):
        model = GaussianProcess(1, self.ker_par_1d, 'rbf', 'ut', self.pt_par_ut)
        y = fcn(model.points)
        lhyp = np.log([1.0, 3.0])
        f, df = model.neg_log_marginal_likelihood(lhyp, y.T, model.points)

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
        res_ml2 = model.optimize(lhyp0, y.T.squeeze(), model.points, method='BFGS', constraints=con)
        # res_ml2_emv = model.optimize(lhyp0, y.T, crit='nlml+emv', method='SLSQP', constraints=con)
        # res_ml2_ivar = model.optimize(lhyp0, y.T, crit='nlml+ivar', method='SLSQP', constraints=con)
        hyp_ml2 = np.exp(res_ml2.x)
        # hyp_ml2_emv = np.exp(res_ml2_emv.x)
        # hyp_ml2_ivar = np.exp(res_ml2_ivar.x)

        print(res_ml2)
        # print(res_ml2_emv)
        # print(res_ml2_ivar)

        print('ML-II({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1]))
        # print('ML-II-EMV({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2_emv.fun, hyp_ml2_emv[0], hyp_ml2_emv[1]))
        # print('ML-II-IVAR({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2_ivar.fun, hyp_ml2_ivar[0],
        #                                                               hyp_ml2_ivar[1]))

        # plot after optimization
        model.plot_model(xtest, y, fcn_true=f, par=hyp_ml2)
        # model.plot_model(xtest, y, fcn_true=f, par=hyp_ml2_emv)
        # model.plot_model(xtest, y, fcn_true=f, par=hyp_ml2_ivar)

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
        from models.tracking import ReentryRadar
        ssm = ReentryRadar()
        func = ssm.dyn_eval
        dim_in, dim_out = ssm.xD, ssm.xD

        model = GaussianProcessMO(dim_in, dim_out, self.ker_par_5d, 'rbf', 'sr')  # , point_hyp={'degree': 10})
        x = ssm.get_pars('x0_mean')[0][:, na] + model.points  # ssm.get_pars('x0_cov')[0].dot(model.points)
        y = np.apply_along_axis(func, 0, x, None)  # (d_out, n**2)

        lhyp0 = np.log(np.ones((dim_out, dim_in+1)))

        res_ml2 = model.optimize(lhyp0, y, model.points, method='BFGS')
        # res_ml2_emv = model.optimize(lhyp0, y.T, crit='nlml+emv', method='L-BFGS-B', bounds=b)
        # res_ml2_ivar = model.optimize(lhyp0, y.T, crit='nlml+ivar', method='L-BFGS-B', bounds=b)
        hyp_ml2 = np.exp(res_ml2.x)
        # hyp_ml2_emv = np.exp(res_ml2_emv.x)
        # hyp_ml2_ivar = np.exp(res_ml2_ivar.x)

        print(res_ml2)
        # print(res_ml2_emv)
        # print(res_ml2_ivar)
        np.set_printoptions(precision=4)
        print('ML-II({:.4f}) @ alpha: {:.4f}, el: {}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1:]))
        # print('ML-II-EMV({:.4f}) @ alpha: {:.4f}, el: {}'.format(res_ml2_emv.fun, hyp_ml2_emv[0], hyp_ml2_emv[1:]))
        # print('ML-II-IVAR({:.4f}) @ alpha: {:.4f}, el: {}'.format(res_ml2_ivar.fun, hyp_ml2_ivar[0],
        #                                                           hyp_ml2_ivar[1:]))


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
        self.assertTrue(model.exp_model_variance(y) >= 0)

    def test_integral_variance(self):
        model = StudentTProcess(1, self.ker_par_1d)
        y = fcn(model.points)
        self.assertTrue(model.integral_variance(y) >= 0)

    @staticmethod
    def _nlml(log_par, kernel, fcn_obs, x_obs, jitter, nu):
        # convert from log-par to par
        par = np.exp(log_par)
        num_data = x_obs.shape[1]

        K = kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, )
        y_dot_a = fcn_obs.T.dot(a)

        # negative marginal log-likelihood
        from scipy.special import gamma
        half_logdet_K = np.sum(np.log(np.diag(L[0])))
        const = 0.5 * num_data * np.log((nu - 2) * np.pi) + np.log(gamma(0.5 * nu + num_data) / gamma(0.5 * nu))

        return 0.5 * (nu + num_data) * np.log(1 + y_dot_a) + half_logdet_K + const

    @staticmethod
    def _nlml_grad(log_par, kernel, fcn_obs, x_obs, jitter, nu):
        # convert from log-par to par
        par = np.exp(log_par)
        num_data = x_obs.shape[1]

        K = kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, )
        y_dot_a = fcn_obs.T.dot(a)
        a_out_a = np.outer(a, a.T)  # (N, N)

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = kernel.der_par(par, x_obs)  # (N, N, num_par)
        iKdK = la.cho_solve(L, dK_dTheta)
        scale = (nu + num_data) / (nu + y_dot_a - 2)

        return 0.5 * np.trace((iKdK - scale * a_out_a.dot(dK_dTheta)))  # (num_par, )

    def test_nlml_gradient(self):
        model = StudentTProcess(5, self.ker_par_5d, 'rbf', 'ut', self.pt_par_ut)
        y = fcn(model.points)
        lhyp = np.log([1.0] + 5 * [3.0])
        jitter = 1e-8 * np.eye(model.num_pts)
        dof = 3

        from scipy.optimize import check_grad
        err = check_grad(self._nlml, self._nlml_grad, lhyp, model.kernel, y.T[:, 0], model.points, jitter, dof)
        print(err)
        self.assertTrue(err <= 1e-5, 'Gradient error: {:.4f}'.format(err))

    def test_hypers_optim(self):
        model = StudentTProcess(1, self.ker_par_1d, 'rbf', 'gh', point_par={'degree': 10})
        xtest = np.linspace(-7, 7, 100)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        # plot before optimization
        model.plot_model(xtest, y, fcn_true=f)
        lhyp0 = np.log([[1.0, 0.5]])
        b = ((np.log(0.9), np.log(1.1)), (None, None))

        def con_alpha(lhyp):
            # constrain alpha**2 = 1
            return np.exp(lhyp[0]) ** 2 - 1 ** 2

        con = {'type': 'eq', 'fun': con_alpha}
        res_ml2 = model.optimize(lhyp0, y.T.squeeze(), model.points, method='BFGS', constraints=con)
        hyp_ml2 = np.exp(res_ml2.x)

        print(res_ml2)

        print('ML-II({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1]))

        # plot after optimization
        model.plot_model(xtest, y, fcn_true=f, par=hyp_ml2)
