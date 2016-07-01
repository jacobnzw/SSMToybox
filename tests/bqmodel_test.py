from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as na
from transforms.bqmodel import GaussianProcess, StudentTProcess

# fcn = lambda x: np.sin((x + 1) ** -1)
fcn = lambda x: 0.5 * x + 25 * x / (1 + x ** 2)


# fcn = lambda x: np.sin(x)
# fcn = lambda x: 0.05*x ** 2
# fcn = lambda x: x


class GPModelTest(TestCase):
    # TODO: could be general test class for any model

    def test_init(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        phyp = {'alpha': 1.0}
        GaussianProcess(1)
        GaussianProcess(1, kernel='rbf', points='ut', kern_hyp=khyp, point_hyp=phyp)

    def test_plotting(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        model.plot_model(xtest, y, fcn_true=f)

    def test_exp_model_variance(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        y = fcn(model.points)
        self.assertTrue(model.exp_model_variance(y) >= 0)

    def test_integral_variance(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        y = fcn(model.points)
        self.assertTrue(model.integral_variance(y) >= 0)

    def test_log_marginal_likelihood(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        y = fcn(model.points)
        lhyp = np.log([1.0, 3.0])
        f, df = model.neg_log_marginal_likelihood(lhyp, y.T)

    def test_hypers_optim(self):
        khyp = {'alpha': 1.0, 'el': 1.0 * np.ones(1)}
        model = GaussianProcess(1, points='gh', kern_hyp=khyp, point_hyp={'degree': 15})
        xtest = np.linspace(-7, 7, 100)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        # plot before optimization
        # model.plot_model(xtest, y, fcn_true=f)
        lhyp0 = np.log([1.0, 0.1])
        b = ((np.log(0.9), np.log(1.1)), (None, None))

        def con_alpha(lhyp):
            # constrain alpha**2 = 1
            return np.exp(lhyp[0]) ** 2 - 1 ** 2

        con = {'type': 'eq', 'fun': con_alpha}
        res_ml2 = model.optimize(lhyp0, y.T, crit='nlml', method='SLSQP', constraints=con)
        res_ml2_emv = model.optimize(lhyp0, y.T, crit='nlml+emv', method='SLSQP', constraints=con)
        res_ml2_ivar = model.optimize(lhyp0, y.T, crit='nlml+ivar', method='SLSQP', constraints=con)
        hyp_ml2 = np.exp(res_ml2.x)
        hyp_ml2_emv = np.exp(res_ml2_emv.x)
        hyp_ml2_ivar = np.exp(res_ml2_ivar.x)

        print res_ml2
        print res_ml2_emv
        print res_ml2_ivar

        print 'ML-II({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1])
        print 'ML-II-EMV({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2_emv.fun, hyp_ml2_emv[0], hyp_ml2_emv[1])
        print 'ML-II-IVAR({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2_ivar.fun, hyp_ml2_ivar[0],
                                                                      hyp_ml2_ivar[1])

        # plot after optimization
        model.plot_model(xtest, y, fcn_true=f, hyp=hyp_ml2)
        model.plot_model(xtest, y, fcn_true=f, hyp=hyp_ml2_emv)
        model.plot_model(xtest, y, fcn_true=f, hyp=hyp_ml2_ivar)

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
        dim = 5
        from models.tracking import ReentryRadar
        ssm = ReentryRadar()
        func = ssm.meas_eval
        khyp = {'alpha': 1.0, 'el': 1.0 * np.ones(dim)}
        model = GaussianProcess(dim, points='sr', kern_hyp=khyp)  # , point_hyp={'degree': 10})
        x = ssm.get_pars('x0_mean')[0][:, na] + model.points  # ssm.get_pars('x0_cov')[0].dot(model.points)
        y = np.apply_along_axis(func, 0, x, None)  # (d_out, n**2)

        lhyp0 = np.log([1.0] + [1000] * dim)
        b = ((np.log(1.0), np.log(1.0)),) + ((None, None),) * dim

        def con_alpha(lhyp):
            # constrain alpha**2 = 1
            return np.exp(lhyp[0]) ** 2 - 1 ** 2

        con = {'type': 'eq', 'fun': con_alpha}
        res_ml2 = model.optimize(lhyp0, y.T, crit='nlml', method='L-BFGS-B', bounds=b)
        res_ml2_emv = model.optimize(lhyp0, y.T, crit='nlml+emv', method='L-BFGS-B', bounds=b)
        res_ml2_ivar = model.optimize(lhyp0, y.T, crit='nlml+ivar', method='L-BFGS-B', bounds=b)
        hyp_ml2 = np.exp(res_ml2.x)
        hyp_ml2_emv = np.exp(res_ml2_emv.x)
        hyp_ml2_ivar = np.exp(res_ml2_ivar.x)

        print res_ml2
        print res_ml2_emv
        print res_ml2_ivar
        np.set_printoptions(precision=4)
        print 'ML-II({:.4f}) @ alpha: {:.4f}, el: {}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1:])
        print 'ML-II-EMV({:.4f}) @ alpha: {:.4f}, el: {}'.format(res_ml2_emv.fun, hyp_ml2_emv[0], hyp_ml2_emv[1:])
        print 'ML-II-IVAR({:.4f}) @ alpha: {:.4f}, el: {}'.format(res_ml2_ivar.fun, hyp_ml2_ivar[0],
                                                                  hyp_ml2_ivar[1:])


class TPModelTest(TestCase):
    def test_init(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        phyp = {'alpha': 1.0}
        StudentTProcess(1)
        StudentTProcess(1, kernel='rbf', points='ut', kern_hyp=khyp, point_hyp=phyp)

    def test_plotting(self):
        dim = 1
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(dim, )}
        model = StudentTProcess(dim, kern_hyp=khyp)
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        model.plot_model(xtest, y, fcn_true=f)

    def test_exp_model_variance(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = StudentTProcess(1, kern_hyp=khyp)
        y = fcn(model.points)
        self.assertTrue(model.exp_model_variance(y) >= 0)

    def test_integral_variance(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = StudentTProcess(1, kern_hyp=khyp)
        y = fcn(model.points)
        self.assertTrue(model.integral_variance(y) >= 0)

    def test_hypers_optim(self):
        khyp = {'alpha': 1.0, 'el': 1.0 * np.ones(1)}
        model = StudentTProcess(1, points='gh', kern_hyp=khyp, point_hyp={'degree': 15})
        xtest = np.linspace(-7, 7, 100)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        # plot before optimization
        # model.plot_model(xtest, y, fcn_true=f)
        lhyp0 = np.log([1.0, 1.0])
        b = ((np.log(0.9), np.log(1.1)), (None, None))

        def con_alpha(lhyp):
            # constrain alpha**2 = 1
            return np.exp(lhyp)[0] ** 2 - 1

        con = {'type': 'eq', 'fun': con_alpha}
        res_ml2 = model.optimize(lhyp0, y.T, crit='nlml', method='BFGS', constraints=con)
        res_ml2_emv = model.optimize(lhyp0, y.T, crit='nlml+emv', method='BFGS', constraints=con)
        res_ml2_ivar = model.optimize(lhyp0, y.T, crit='nlml+ivar', method='BFGS', constraints=con)
        hyp_ml2 = np.exp(res_ml2.x)
        hyp_ml2_emv = np.exp(res_ml2_emv.x)
        hyp_ml2_ivar = np.exp(res_ml2_ivar.x)

        print res_ml2
        print res_ml2_emv
        print res_ml2_ivar

        print 'ML-II({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2.fun, hyp_ml2[0], hyp_ml2[1])
        print 'ML-II-EMV({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2_emv.fun, hyp_ml2_emv[0], hyp_ml2_emv[1])
        print 'ML-II-IVAR({:.4f}) @ alpha: {:.4f}, el: {:.4f}'.format(res_ml2_ivar.fun, hyp_ml2_ivar[0],
                                                                      hyp_ml2_ivar[1])

        # plot after optimization
        model.plot_model(xtest, y, fcn_true=f, hyp=hyp_ml2)
        model.plot_model(xtest, y, fcn_true=f, hyp=hyp_ml2_emv)
        model.plot_model(xtest, y, fcn_true=f, hyp=hyp_ml2_ivar)
