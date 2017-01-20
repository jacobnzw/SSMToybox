from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na
from numpy.linalg import cholesky

from .mtform import MomentTransform

# TODO: docstrings


class BQTransform(MomentTransform, metaclass=ABCMeta):

    # list of supported models for the integrand
    _supported_models_ = ['gp', 'gp-mo', 'tp']  # mgp, gpder, ...

    def __init__(self, dim_in, kern_hyp, dim_out=1, model='gp', kernel='rbf', points='ut', point_par=None, **kwargs):
        self.model = BQTransform._get_model(dim_in, dim_out, model, kernel, points, kern_hyp, point_par)

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
    def _get_model(dim_in, dim_out, model, kernel, points, hypers, point_pars, **kwargs):
        """

        Parameters
        ----------
        dim_in : int
        dim_out : int
        model : string
        kernel : string
        points : string
        hypers : numpy.ndarray
        point_pars : numpy.ndarray
        kwargs : dict

        Returns
        -------
        : Model

        """

        # import must be after SigmaPointTransform
        from .bqmodel import GaussianProcess, StudentTProcess, GaussianProcessMO
        model = model.lower()

        # make sure kernel is supported
        if model not in BQTransform._supported_models_:
            print('Model {} not supported. Supported models are {}.'.format(model, BQTransform._supported_models_))
            return None

        # initialize the chosen model
        if model == 'gp':
            return GaussianProcess(dim_in, hypers, kernel, points, point_pars)
        elif model == 'gp-mo':
            return GaussianProcessMO(dim_in, dim_out, hypers, kernel, points, point_pars)
        elif model == 'tp':
            return StudentTProcess(dim_in, hypers, kernel, points, point_pars)

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
        expected_model_var = self.model.emv
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _cross_covariance(self, weights, fcn_evals, chol_cov_in):
        return fcn_evals.dot(weights.T).dot(chol_cov_in.T)

    def __str__(self):
        return '{}\n{}'.format(self.__class__.__name__, self.model)


class GPQ(BQTransform):  # consider renaming to GPQTransform
    def __init__(self, dim_in, kern_hyp, kernel='rbf', points='ut', point_par=None):
        super(GPQ, self).__init__(dim_in, 1, kern_hyp, 'gp', kernel, points, point_par)

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


class GPQMO(BQTransform):
    def __init__(self, dim_in, dim_out, kern_hyp, kernel='rbf', points='ut', point_par=None):
        super(GPQMO, self).__init__(dim_in, kern_hyp, dim_out, 'gp-mo', kernel, points, point_par)

        # output dimension (number of outputs)
        self.e = dim_out

    def _weights(self, tf_pars=None):
        """
        Weights of the multi-output Gaussian process quadrature.

        Parameters
        ----------
        tf_pars : numpy.ndarray of shape (E, num_par)
            Kernel parameters in a matrix, where e-th row contains parameters for e-th output.

        Returns
        -------
        : tuple (w_m, w_c, w_cc)
            GP quadrature weights for the mean (w_m), covariance (w_c) and cross-covariance (w_cc).
            w_m : numpy.ndarray of shape (N, E)
            w_c : numpy.ndarray of shape (N, N, E, E)
            w_cc : numpy.ndarray of shape (D, N, E)

        """

        # if tf_pars=None return parameters stored in Kernel
        par = self.model.kernel.get_hyperparameters(tf_pars)

        # retrieve sigma-points from Model
        x = self.model.points
        d, e, n = self.model.dim_in, self.model.dim_out, self.model.num_pts

        # Kernel expectations
        q = np.zeros((n, e))
        Q = np.zeros((n, n, e, e))
        R = np.zeros((d, n, e))
        iK = np.zeros((n, n, e))
        for i in range(self.e):
            q[:, i] = self.model.kernel.exp_x_kx(x, par[i, :])
            R[..., i] = self.model.kernel.exp_x_xkx(x, par[i, :])
            iK[..., i] = self.model.kernel.eval_inv_dot(x, par[i, :], scaling=True)
            for j in range(self.e):
                Q[..., i, j] = self.model.kernel.exp_x_kxkx(x, par[i, :], par[j, :])

        # weights
        # w_m = q iK
        # w_c = iK Q iK
        # w_cc = R iK

    def _fcn_eval(self, fcn, x, fcn_pars):
        return np.apply_along_axis(fcn, 0, x, fcn_pars)

    def _mean(self, weights, fcn_evals):
        return (fcn_evals * weights).sum(axis=0)

    def _covariance(self, weights, fcn_evals, mean_out):
        pass

    def _cross_covariance(self, weights, fcn_evals, chol_cov_in):
        pass

    def _integral_variance(self, points, tf_pars):
        pass


class TPQ(BQTransform):
    def __init__(self, dim_in, kern_hyp, kernel='rbf', points='ut', point_par=None):
        super(TPQ, self).__init__(dim_in, kern_hyp, model='tp', kernel=kernel, points=points, point_par=point_par)

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

    def _covariance(self, weights, fcn_evals, mean_out):
        expected_model_var = self.model.exp_model_variance(fcn_evals)
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _integral_variance(self, points, tf_pars):
        pass
