from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na
from numpy.linalg import cholesky

from ssmtoybox.mtran import MomentTransform


class BQTransform(MomentTransform, metaclass=ABCMeta):
    """
    Base class for Bayesian Quadrature moment transforms.

    Parameters
    ----------
    dim_in : int
        Dimensionality of the input.

    dim_out : int
        Dimensionality of the output.

    kern_par : ndarray
        Kernel parameters.

    model : str {'gp', 'tp', 'gp-mo', 'tp-mo'}
        Probabilistic model of the integrand.
        'gp'
            Gaussian process.
        'tp'
            Student's t-process.
        'gp-mo'
            Multi-output Gaussian process.
        'tp-mo'
            Multi-output Student's t-process.

    kernel : str {'rbf'}
        Kernel of the integrand model.

    points : str {'ut', 'sr', 'gh', 'fs'}
        Sigma-point set for representing the input probability density.

    point_par : dict
        Sigma-point set parameters.

    Attributes
    ----------
    BQTransform._supported_models_ : list ['gp', 'gp-mo', 'tp', 'tp-mo']
        The implemented probabilistic models of the integrand.
    """

    # list of supported models for the integrand
    _supported_models_ = ['gp', 'gp-mo', 'tp', 'tp-mo', 'bs']

    def __init__(self, dim_in, dim_out, kern_par, model, kern_str, point_str, point_par, **kwargs):
        self.model = BQTransform._get_model(dim_in, dim_out, model, kern_str, point_str, kern_par, point_par, **kwargs)
        self.I_out = np.eye(dim_out)

    def apply(self, f, mean, cov, fcn_par, kern_par=None):
        """
        Compute transformed moments.

        Parameters
        ----------
        f : func
            Integrand, transforming function of the random variable.

        mean : ndarray
            Input mean.

        cov : ndarray
            Input covariance.

        fcn_par : ndarray
            Integrand parameters.

        kern_par : ndarray
            Kernel parameters.

        Returns
        -------
        mean_f : ndarray
            Transformed mean.

        cov_f : ndarray
            Transformed covariance.

        cov_fx : ndarray
            Covariance between input and output random variables.
        """

        # re-compute weights if transform parameter kern_par explicitly given
        if kern_par is not None:
            self.wm, self.Wc, self.Wcc = self.weights(kern_par)

        mean = mean[:, na]
        chol_cov = cholesky(cov)

        # evaluate integrand at sigma-points
        x = mean + chol_cov.dot(self.model.points)
        fx = self._fcn_eval(f, x, fcn_par)

        # compute transformed moments
        mean_f = self._mean(self.wm, fx)
        cov_f = self._covariance(self.Wc, fx, mean_f)
        cov_fx = self._cross_covariance(self.Wcc, fx, chol_cov)

        return mean_f, cov_f, cov_fx

    def weights(self, par, *args):
        """
        Bayesian quadrature weights.

        Parameters
        ----------
        par : ndarray
            Kernel parameters to use in computation of the weights.

        args : tuple
            Other relevant parameters.

        Returns
        -------
        : tuple
            Weights for the mean, covariance and cross-covariance quadrature approximations.

        """
        wm, wc, wcc, emv, ivar = self.model.bq_weights(par, *args)
        return wm, wc, wcc

    def _fcn_eval(self, fcn, x, fcn_par):
        """
        Evaluations of the integrand, which can comprise function observations as well as derivative observations.

        Parameters
        ----------
        fcn : func
            Integrand as a function handle, which is expected to behave certain way.

        x : ndarray
            Argument (input) of the integrand.

        fcn_par :
            Parameters of the integrand.

        Returns
        -------
        : ndarray
            Function evaluations of shape (out_dim, num_pts).

        Notes
        -----
        Methods in derived subclasses decides whether to return derivatives also
        """
        return np.apply_along_axis(fcn, 0, x, fcn_par)

    def _mean(self, weights, fcn_evals):
        """
        Transformed mean.

        Parameters
        ----------
        weights : ndarray
            Quadrature weights.

        fcn_evals : ndarray
            Integrand evaluations.

        Returns
        -------
        : ndarray
            Transformed mean.
        """
        return fcn_evals.dot(weights)
        # return np.einsum('en, n -> e', fcn_evals, weights)

    def _covariance(self, weights, fcn_evals, mean_out):
        """
        Transformed covariance.

        Parameters
        ----------
        weights : ndarray
            Quadrature weights.

        fcn_evals : ndarray
            Integrand evaluations.

        mean_out : ndarray
            Transformed mean.

        Returns
        -------
        : ndarray
            Transformed covariance.
        """
        emv = self.model.model_var*self.I_out
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + emv
        # return np.einsum('in, nm, jm -> ij', fcn_evals, weights, fcn_evals) - \
        #        np.outer(mean_out, mean_out.T) + expected_model_var

    def _cross_covariance(self, weights, fcn_evals, chol_cov_in):
        """
        Cross-covariance of input variable and transformed output variable.

        Parameters
        ----------
        weights : (D, N) ndarray
            Quadrature weights.

        fcn_evals : (E, N) ndarray
            Integrand evaluations.

        chol_cov_in : (D, D) ndarray
            Cholesky factor of the input covariance.

        Returns
        -------
        : ndarray
            Covariance between the input and transformed random variable.
        """
        return fcn_evals.dot(weights.T).dot(chol_cov_in.T)
        # return np.einsum('en, dn, dj -> ej', fcn_evals, weights, chol_cov_in)

    @staticmethod
    def _get_model(dim_in, dim_out, model, kern_str, point_str, kern_par, point_par, **kwargs):
        """
        Initialize chosen model with supplied parameters.

        Parameters
        ----------
        dim_in : int
            Input dimensionality.

        dim_out : int
            Output dimensionality.

        model : str
            Model of the integrand. See `BQTransform._supported_models_`.

        kern_str : str
            Kernel of the model. See `Model._supported_kernels_`.

        point_str : str
            Point-set to use for the integration. See `Model._supported_points_`.

        kern_par : ndarray
            Kernel parameters.

        point_par : dict
            Parameters of the point-set scheme.

        kwargs : dict
            Additional kwargs passed to the model.

        Returns
        -------
        : Model
            Initialized model.
        """

        # import must be after SigmaPointTransform
        from .bqmod import GaussianProcess, StudentTProcess, GaussianProcessMO, StudentTProcessMO, BayesSardModel
        model = model.lower()

        # make sure kern_str is supported
        if model not in BQTransform._supported_models_:
            print('Model {} not supported. Supported models are {}.'.format(model, BQTransform._supported_models_))
            return None

        # initialize the chosen model
        if model == 'gp':
            return GaussianProcess(dim_in, kern_par, kern_str, point_str, point_par)
        elif model == 'tp':
            return StudentTProcess(dim_in, kern_par, kern_str, point_str, point_par, **kwargs)
        elif model == 'gp-mo':
            return GaussianProcessMO(dim_in, dim_out, kern_par, kern_str, point_str, point_par)
        elif model == 'tp-mo':
            return StudentTProcessMO(dim_in, dim_out, kern_par, kern_str, point_str, point_par, **kwargs)
        elif model == 'bs':
            return BayesSardModel(dim_in, kern_par, point_str=point_str, point_par=point_par, **kwargs)

    def __str__(self):
        return '{}\n{}'.format(self.__class__.__name__, self.model)


class GaussianProcessTransform(BQTransform):
    """
    Gaussian process quadrature moment transform.

    Parameters
    ----------
    dim_in : int
        Dimensionality of the input.

    kern_par : ndarray
        Kernel parameters.

    kern_str : str {'rbf'}
        Kernel of the integrand model.

    point_str : str {'ut', 'sr', 'gh', 'fs'}
        Sigma-point set for representing the input probability density.

    point_par : dict
        Sigma-point set parameters.
    """
    def __init__(self, dim_in, kern_par, kern_str='rbf', point_str='ut', point_par=None):
        super(GaussianProcessTransform, self).__init__(dim_in, 1, kern_par, 'gp', kern_str, point_str, point_par)
        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self.weights(kern_par)


class BayesSardTransform(BQTransform):
    """
    Bayes-Sard quadrature moment transform.

    Parameters
    ----------
    dim_in : int
        Dimensionality of the input.

    kern_par : ndarray
        Kernel parameters.

    multi_ind : int or ndarray, optional
        Multi-index.

    point_str : str {'ut', 'sr', 'gh', 'fs'}, optional
        Sigma-point set for representing the input probability density.

    point_par : dict, optional
        Sigma-point set parameters.
    """

    def __init__(self, dim_in, kern_par, multi_ind=2, point_str='ut', point_par=None):
        super(BayesSardTransform, self).__init__(dim_in, 1, kern_par, 'bs', 'rbf', point_str, point_par,
                                                 multi_ind=multi_ind)
        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self.weights(kern_par, multi_ind)

    def weights(self, par, *args):
        """
        Bayesian quadrature weights.

        Parameters
        ----------
        par : ndarray
            Kernel parameters to use in computation of the weights.

        args[0] : ndarray
            Multi-index.

        Returns
        -------
        : tuple
            Weights for the mean, covariance and cross-covariance quadrature approximations.

        """
        multi_ind = args[0]
        wm, wc, wcc, emv, ivar = self.model.bq_weights(par, multi_ind)
        return wm, wc, wcc


class StudentTProcessTransform(BQTransform):
    """
    Student's t-process quadrature moment transforms.

    Parameters
    ----------
    dim_in : int
        Dimensionality of the input.

    kern_par : ndarray
        Kernel parameters.

    kern_str : str {'rbf'}, optional
        Kernel of the integrand model.

    point_str : str {'ut', 'sr', 'gh', 'fs'}, optional
        Sigma-point set for representing the input probability density.

    point_par : None or dict, optional
        Sigma-point set parameters.

    nu : float
        Degrees of freedom parameter of the t-process regression model.
    """
    def __init__(self, dim_in, kern_par, kern_str='rbf', point_str='ut', point_par=None, nu=3.0):
        super(StudentTProcessTransform, self).__init__(dim_in, 1, kern_par, 'tp', kern_str, point_str, point_par, nu=nu)
        # BQ transform weights for the mean, covariance and cross-covariance
        self.wm, self.Wc, self.Wcc = self.weights(kern_par)

    def _covariance(self, weights, fcn_evals, mean_out):
        """
        TPQ transformed covariance.

        Parameters
        ----------
        weights : ndarray
            Quadrature weights.

        fcn_evals : ndarray
            Integrand evaluations.

        mean_out : ndarray
            Transformed mean.

        Returns
        -------
        : ndarray
            Transformed covariance.
        """
        emv = self.model.exp_model_variance(self.model.kernel.par, fcn_evals) * self.I_out
        return fcn_evals.dot(weights).dot(fcn_evals.T) - np.outer(mean_out, mean_out.T) + emv


class GPQMO(BQTransform):
    def __init__(self, dim_in, dim_out, kern_par, kern_str='rbf', point_str='ut', point_par=None):
        """

        Parameters
        ----------
        dim_in
        dim_out
        kern_par
        kern_str
        point_str
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
        super(GPQMO, self).__init__(dim_in, dim_out, kern_par, 'gp-mo', kern_str, point_str, point_par)

        # output dimension (number of outputs)
        self.e = dim_out

        # TEMPORARY
        self.tmean = np.empty((dim_out, ))
        self.tcov = np.empty((dim_out, dim_out))
        self.tccov = np.empty((dim_out, dim_in))

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


class TPQMO(BQTransform):

    def __init__(self, dim_in, dim_out, kern_par, kern_str='rbf', point_str='ut', point_par=None, nu=3.0):

        super(TPQMO, self).__init__(dim_in, dim_out, kern_par, 'tp-mo', kern_str, point_str, point_par, nu=nu)

        # output dimension (number of outputs)
        self.e = dim_out

        # TEMPORARY
        self.tmean = np.empty((dim_out,))
        self.tcov = np.empty((dim_out, dim_out))
        self.tccov = np.empty((dim_out, dim_in))

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
