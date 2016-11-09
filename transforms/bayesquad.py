from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na
from numpy.linalg import cholesky

from .mtform import MomentTransform


class BQTransform(MomentTransform, metaclass=ABCMeta):
    _supported_models_ = ['gp', 'tp']  # mgp, gpder, ...

    def __init__(self, dim, model='gp', kernel=None, points=None, kern_hyp=None, point_par=None, **kwargs):
        self.model = BQTransform._get_model(dim, model, kernel, points, kern_hyp, point_par, **kwargs)
        self.d, self.n = self.model.points.shape
        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self._weights()

    def apply(self, f, mean, cov, fcn_pars, tf_pars=None):
        # Re-compute weights if transform parameter tf_pars explicitly given
        if tf_pars is not None:
            self.wm, self.Wc, self.Wcc = self._weights(tf_pars)
        mean = mean[:, na]
        chol_cov = cholesky(cov)
        x = mean + chol_cov.dot(self.model.points)
        fx = self._fcn_eval(f, x, fcn_pars)
        mean_f = self._mean(self.wm, fx)
        cov_f = self._covariance(self.Wc, fx, mean_f)
        cov_fx = self._cross_covariance(self.Wcc, fx, chol_cov)
        return mean_f, cov_f, cov_fx

    @staticmethod
    def _get_model(dim, model, kernel, points, hypers, point_pars, **kwargs):
        from .bqmodel import GaussianProcess, StudentTProcess  # import must be after SigmaPointTransform
        model = model.lower()
        # make sure kernel is supported
        if model not in BQTransform._supported_models_:
            print('Model {} not supported. Supported models are {}.'.format(model, BQTransform._supported_models_))
            return None
        # initialize the chosen model
        if model == 'gp':
            return GaussianProcess(dim, kernel, points, hypers, point_pars, **kwargs)
        elif model == 'tp':
            return StudentTProcess(dim, kernel, points, hypers, point_pars, **kwargs)

    # TODO: specify requirements for shape of input/output for all of these fcns

    def minimum_variance_points(self, x0, tf_pars):
        # run optimizer to find minvar point sets using initial guess x0; requires implemented _integral_variance()
        pass

    @abstractmethod
    def _weights(self, tf_pars):
        # no need for input args because points and hypers are in self.model.points and self.model.kernel.hypers
        pass

    @abstractmethod
    def _integral_variance(self, points, tf_pars):
        # can serve for finding minimum variance point sets or hyper-parameters
        # optimizers require the first argument to be the variable, a decorator could be used to interchange the first
        # two arguments, so that we don't have to define the same function twice only w/ different signature
        pass

    @abstractmethod
    def _fcn_eval(self, fcn, x, fcn_pars):
        # derived class decides whether to return derivatives also
        pass

    def _mean(self, weights, fcn_evals):
        return fcn_evals.dot(weights)

    def _covariance(self, weights, fcn_evals, mean_out):
        # TODO: pre-compute EMV during init, when using TPQ multiply by factor based on fcn_evals
        expected_model_var = self.model.exp_model_variance(fcn_evals)
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _cross_covariance(self, weights, fcn_evals, chol_cov_in):
        return fcn_evals.dot(weights.T).dot(chol_cov_in.T)

    def __str__(self):
        return '{}\n{}'.format(self.__class__.__name__, self.model)


class GPQ(BQTransform):  # consider renaming to GPQTransform
    def __init__(self, dim, kernel, points, kern_hyp=None, point_par=None):
        super(GPQ, self).__init__(dim, 'gp', kernel, points, kern_hyp, point_par)

    def _weights(self, tf_pars=None):
        x = self.model.points
        iK = self.model.kernel.eval_inv_dot(x, tf_pars, ignore_alpha=True)

        # Kernel expectations
        q = self.model.kernel.exp_x_kx(x, tf_pars)
        Q = self.model.kernel.exp_x_kxkx(x, tf_pars)
        R = self.model.kernel.exp_x_xkx(x, tf_pars)

        # BQ weights in terms of kernel expectations
        w_m = q.dot(iK)
        w_c = iK.dot(Q).dot(iK)
        w_cc = R.dot(iK)
        return w_m, w_c, w_cc

    def _fcn_eval(self, fcn, x, fcn_pars):
        return np.apply_along_axis(fcn, 0, x, fcn_pars)

    def _integral_variance(self, points, tf_pars):
        pass


class TPQ(BQTransform):
    def __init__(self, dim, kernel, points, kern_hyp=None, point_par=None, nu=None):
        super(TPQ, self).__init__(dim, 'tp', kernel, points, kern_hyp, point_par, nu=nu)

    def _weights(self, tf_pars=None):
        x = self.model.points
        iK = self.model.kernel.eval_inv_dot(x, tf_pars, ignore_alpha=True)

        # Kernel expectations
        q = self.model.kernel.exp_x_kx(x, tf_pars)
        Q = self.model.kernel.exp_x_kxkx(x, tf_pars)
        R = self.model.kernel.exp_x_xkx(x, tf_pars)

        # BQ weights in terms of kernel expectations
        w_m = q.dot(iK)
        w_c = iK.dot(Q).dot(iK)
        w_cc = R.dot(iK)
        return w_m, w_c, w_cc

    def _fcn_eval(self, fcn, x, fcn_pars):
        return np.apply_along_axis(fcn, 0, x, fcn_pars)

    def _integral_variance(self, points, tf_pars):
        pass
