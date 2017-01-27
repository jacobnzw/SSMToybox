from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na
from numpy.linalg import cholesky

from .mtform import MomentTransform

# TODO: docstrings


class BQTransform(MomentTransform, metaclass=ABCMeta):

    # list of supported models for the integrand
    _supported_models_ = ['gp', 'gp-mo', 'tp']  # mgp, gpder, ...

    def __init__(self, dim_in, dim_out, kern_par, model, kernel, points, point_par, **kwargs):
        self.model = BQTransform._get_model(dim_in, dim_out, model, kernel, points, kern_par, point_par)

        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self._weights()

    def apply(self, f, mean, cov, fcn_par, kern_par=None):
        """
        Compute transformed moments.

        Parameters
        ----------
        f : func
            Integrand, transforming function of the random variable.
        mean : numpy.ndarray
            Input mean.
        cov : numpy.ndarray
            Input covariance.
        fcn_par : numpy.ndarray
            Integrand parameters.
        kern_par : numpy.ndarray
            Kernel parameters.

        Returns
        -------
        : tuple
            Transformed mean, covariance and cross-covariance in tuple.

        """

        # re-compute weights if transform parameter kern_par explicitly given
        if kern_par is not None:
            self.wm, self.Wc, self.Wcc = self._weights(kern_par)

        mean = mean[:, na]
        chol_cov = cholesky(cov)

        # evaluate integrand at sigma-points
        x = mean + chol_cov.dot(self.model.points)
        fx = self._fcn_eval(f, x, fcn_par)

        # DEBUG
        self.fx = fx

        # compute transformed moments
        mean_f = self._mean(self.wm, fx)
        cov_f = self._covariance(self.Wc, fx, mean_f)
        cov_fx = self._cross_covariance(self.Wcc, fx, chol_cov)

        return mean_f, cov_f, cov_fx

    @staticmethod
    def _get_model(dim_in, dim_out, model, kernel, points, kern_par, point_par, **kwargs):
        """
        Initialize chosen model with supplied parameters.

        Parameters
        ----------
        dim_in : int
        dim_out : int
        model : string
            Model of the integrand. See `BQTransform._supported_models_`.
        kernel : string
            Kernel of the model. See `Model._supported_kernels_`.
        points : string
            Point-set to use for the integration. See `Model._supported_points_`.
        kern_par : numpy.ndarray
            Kernel parameters.
        point_par : dict
            Parameters of the point-set scheme.
        kwargs : dict
            Additional kwargs passed to the model.

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
            return GaussianProcess(dim_in, kern_par, kernel, points, point_par)
        elif model == 'gp-mo':
            return GaussianProcessMO(dim_in, dim_out, kern_par, kernel, points, point_par)
        elif model == 'tp':
            return StudentTProcess(dim_in, kern_par, kernel, points, point_par, **kwargs)

    def minimum_variance_points(self, x0, kern_par):
        # run optimizer to find minvar point sets using initial guess x0; requires implemented _integral_variance()
        pass

    @abstractmethod
    def _weights(self, kern_par):
        """
        Bayesian quadrature weights.

        Parameters
        ----------
        kern_par : numpy.ndarray
            Kernel parameters to use in computation of the weights.

        Returns
        -------
        : tuple
            Weights for the mean, covariance and cross-covariance quadrature approximations.

        """
        pass

    @abstractmethod
    def _integral_variance(self, points, kern_par):
        # can serve for finding minimum variance point sets or kernel parameters
        # optimizers require the first argument to be the variable, a decorator could be used to interchange the first
        # two arguments, so that we don't have to define the same function twice only w/ different signature
        pass

    @abstractmethod
    def _fcn_eval(self, fcn, x, fcn_par):
        """
        Evaluations of the integrand, which can comprise function observations as well as derivative observations.

        Parameters
        ----------
        fcn : func
            Integrand as a function handle, which is expected to behave certain way.
        x : numpy.ndarray
            Argument (input) of the integrand.
        fcn_par :
            Parameters of the integrand.
        Notes
        -----
        Methods in derived subclasses decides whether to return derivatives also

        Returns
        -------

        """
        pass

    def _mean(self, weights, fcn_evals):
        """
        Transformed mean.

        Parameters
        ----------
        weights : numpy.ndarray
        fcn_evals : numpy.ndarray

        Returns
        -------
        : numpy.ndarray

        """
        return fcn_evals.dot(weights)
        # return np.einsum('en, n -> e', fcn_evals, weights)

    def _covariance(self, weights, fcn_evals, mean_out):
        """
        Transformed covariance.

        Parameters
        ----------
        weights : numpy.ndarray
        fcn_evals : numpy.ndarray
        mean_out : numpy.ndarray

        Returns
        -------
        : numpy.ndarray

        """
        expected_model_var = self.model.exp_model_variance(fcn_evals)
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var
        # return np.einsum('in, nm, jm -> ij', fcn_evals, weights, fcn_evals) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _cross_covariance(self, weights, fcn_evals, chol_cov_in):
        """
        Cross-covariance of input variable and transformed output variable.

        Parameters
        ----------
        weights : numpy.ndarray
            Shape (D, N)
        fcn_evals : numpy.ndarray
            Shape (E, N)
        chol_cov_in : numpy.ndarray
            Shape (D, D)

        Returns
        -------
        : numpy.ndarray

        """
        return fcn_evals.dot(weights.T).dot(chol_cov_in.T)
        # return np.einsum('en, dn, dj -> ej', fcn_evals, weights, chol_cov_in)

    def __str__(self):
        return '{}\n{}'.format(self.__class__.__name__, self.model)


class GPQ(BQTransform):  # consider renaming to GPQTransform
    def __init__(self, dim_in, kern_par, kernel='rbf', points='ut', point_par=None):
        super(GPQ, self).__init__(dim_in, 1, kern_par, 'gp', kernel, points, point_par)

    def _weights(self, kern_par=None):
        """
        Weights of the Gaussian process quadrature.

        Parameters
        ----------
        kern_par : array_like

        Returns
        -------
        : tuple
        Weights for computation of the transformed mean, covariance and cross-covariance in a tuple ``(wm, Wc, Wcc)``.

        """
        par = self.model.kernel.get_parameters(kern_par)
        x = self.model.points

        # inverse kernel matrix
        iK = self.model.kernel.eval_inv_dot(par, x, scaling=False)

        # Kernel expectations
        q = self.model.kernel.exp_x_kx(par, x)
        Q = self.model.kernel.exp_x_kxkx(par, par, x)
        R = self.model.kernel.exp_x_xkx(par, x)

        # DEBUG
        self.q, self.Q, self.R, self.iK = q, Q, R, iK

        # BQ weights in terms of kernel expectations
        w_m = q.dot(iK)
        w_c = iK.dot(Q).dot(iK)
        w_cc = R.dot(iK)

        # covariance weights should be symmetric
        w_c = 0.5 * (w_c + w_c.T)

        return w_m, w_c, w_cc

    def _fcn_eval(self, fcn, x, fcn_par):
        return np.apply_along_axis(fcn, 0, x, fcn_par)

    def _integral_variance(self, points, kern_par):
        pass


class GPQMO(BQTransform):
    def __init__(self, dim_in, dim_out, kern_par, kernel='rbf', points='ut', point_par=None):
        """

        Parameters
        ----------
        dim_in
        dim_out
        kern_par
        kernel
        points
        point_par

        Notes
        -----
        There can be a difference between the GPQ weights and GPQMO weights. In GPQMO we need to multiply
        higher-dimensional arrays when computing weights, which is more elegantly handled by the numpy.einsum(). As
        of Numpy 1.11.3 the problem is, that the results from `einsum()` do not exactly match those obtained by
        equivalent usage of `numpy.dot()`. As of yet I did not come up with an implementation giving equal results.

        The transformed moments are implemented in two ways: (a) using `einsum` and (b) using for loops and `dot` (
        slower). The option (b) is the closest to the results produced by GPQ.

        The consequence is that the GPQMO filters do not perform the same as  GPQ filters, even though they should
        when provided with the same parameters.

        """
        super(GPQMO, self).__init__(dim_in, dim_out, kern_par, 'gp-mo', kernel, points, point_par)

        # output dimension (number of outputs)
        self.e = dim_out

        # TEMPORARY
        self.tmean = np.empty((dim_out, ))
        self.tcov = np.empty((dim_out, dim_out))
        self.tccov = np.empty((dim_out, dim_in))

    def _weights(self, kern_par=None):
        """
        Weights of the multi-output Gaussian process quadrature.

        Parameters
        ----------
        kern_par : numpy.ndarray of shape (dim_out, num_par)
            Kernel parameters in a matrix, where e-th row contains parameters for e-th output.

        Returns
        -------
        : tuple (w_m, w_c, w_cc)
            GP quadrature weights for the mean (w_m), covariance (w_c) and cross-covariance (w_cc).
            w_m : numpy.ndarray of shape (num_pts, dim_out)
            w_c : numpy.ndarray of shape (num_pts, num_pts, dim_out, dim_out)
            w_cc : numpy.ndarray of shape (dim_in, num_pts, dim_out)

        """

        # if kern_par=None return parameters stored in Kernel
        par = self.model.kernel.get_parameters(kern_par)

        # retrieve sigma-points from Model
        x = self.model.points
        d, e, n = self.model.dim_in, self.model.dim_out, self.model.num_pts

        # Kernel expectations
        q = np.zeros((n, e))
        Q = np.zeros((n, n, e, e))
        R = np.zeros((d, n, e))
        iK = np.zeros((n, n, e))
        w_c = np.zeros((n, n, e, e))
        for i in range(e):
            q[:, i] = self.model.kernel.exp_x_kx(par[i, :], x)
            R[..., i] = self.model.kernel.exp_x_xkx(par[i, :], x)
            iK[..., i] = self.model.kernel.eval_inv_dot(par[i, :], x, scaling=False)
            for j in range(i+1):
                Q[..., i, j] = self.model.kernel.exp_x_kxkx(par[i, :], par[j, :], x)
                Q[..., j, i] = Q[..., i, j]
                w_c[..., i, j] = iK[..., i].dot(Q[..., i, j]).dot(iK[..., j])
                w_c[..., j, i] = w_c[..., i, j]

        # DEBUG, la.cond(Q) is high
        self.q, self.Q, self.R, self.iK = q, Q, R, iK

        # weights
        # w_m = q(\theta_e) * iK(\theta_e) for all e = 1, ..., dim_out
        w_m = np.einsum('ne, nme -> me', q, iK)

        # w_c = iK(\theta_e) * Q(\theta_e, \theta_f) * iK(\theta_f) for all e,f = 1, ..., dim_out
        # NOTE: einsum gives slighly different results than dot, or I don't know how to use it
        # w_c = np.einsum('nie, ijed, jmd -> nmed', iK, Q, iK)

        # w_cc = R(\theta_e) * iK(\theta_e) for all e = 1, ..., dim_out
        w_cc = np.einsum('die, ine -> dne', R, iK)

        # covariance weights should be symmetric
        w_c = 0.5 * (w_c + w_c.swapaxes(0, 1).swapaxes(2, 3))

        return w_m, w_c, w_cc

    def _fcn_eval(self, fcn, x, fcn_par):
        return np.apply_along_axis(fcn, 0, x, fcn_par)

    def _mean(self, weights, fcn_evals):
        """
        Transformed mean for the multi-output GPQ.

        Parameters
        ----------
        weights : numpy.ndarray
        fcn_evals : numpy.ndarray

        Notes
        -----
        Problems with implementation. Can't get the results to match the results of single-output GPQ transform.
        I strongly suspect this is caused by the inconsistent results from numpy.einsum and numpy.dot.

        Returns
        -------
        : numpy.ndarray

        """
        # return np.einsum('ij, ji -> i', fcn_evals, weights)
        for i in range(self.model.dim_out):
            self.tmean[i] = fcn_evals[i, :].dot(weights[:, i])
        return self.tmean

    def _covariance(self, weights, fcn_evals, mean_out):
        emv = self.model.exp_model_variance(fcn_evals)
        # return np.einsum('ei, ijed, dj -> ed', fcn_evals, weights, fcn_evals) - np.outer(mean_out, mean_out.T) + emv
        for i in range(self.e):
            for j in range(i+1):
                self.tcov[i, j] = fcn_evals[i, :].dot(weights[..., i, j]).dot(fcn_evals[j, :])
                self.tcov[j, i] = self.tcov[i, j]
        return self.tcov - np.outer(mean_out, mean_out.T) + emv

    def _cross_covariance(self, weights, fcn_evals, chol_cov_in):
        """
        Covariance of the input and output variables for multi-output GPQ model.

        Parameters
        ----------
        weights : numpy.ndarray
            Shape (D, N, E)
        fcn_evals : numpy.ndarray
            Shape (E, N)
        chol_cov_in : numpy.ndarray
            Shape (D, D)

        Returns
        -------
        : numpy.ndarray
            Shape (E, D)
        """
        # return np.einsum('en, ine, id  -> ed', fcn_evals, weights, chol_cov_in)
        for i in range(self.e):
            self.tccov[i, :] = fcn_evals[i, :].dot(weights[..., i].T).dot(chol_cov_in.T)
        return self.tccov

    def _integral_variance(self, points, kern_par):
        pass


class TPQ(BQTransform):
    def __init__(self, dim_in, kern_par, kernel='rbf', points='ut', point_par=None):
        super(TPQ, self).__init__(dim_in, 1, kern_par, 'tp', kernel, points, point_par)

    def _weights(self, kern_par=None):
        x = self.model.points
        iK = self.model.kernel.eval_inv_dot(kern_par, x, scaling=False)

        # Kernel expectations
        q = self.model.kernel.exp_x_kx(kern_par, x)
        Q = self.model.kernel.exp_x_kxkx(kern_par, x)
        R = self.model.kernel.exp_x_xkx(kern_par, x)

        # BQ weights in terms of kernel expectations
        w_m = q.dot(iK)
        w_c = iK.dot(Q).dot(iK)
        w_cc = R.dot(iK)
        return w_m, w_c, w_cc

    def _fcn_eval(self, fcn, x, fcn_par):
        return np.apply_along_axis(fcn, 0, x, fcn_par)

    def _covariance(self, weights, fcn_evals, mean_out):
        expected_model_var = self.model.exp_model_variance(fcn_evals)
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + expected_model_var

    def _integral_variance(self, points, kern_par):
        pass
