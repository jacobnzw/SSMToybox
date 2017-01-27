from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize

from .bqkernel import RBF
from .quad import SphericalRadial, Unscented, GaussHermite


# TODO: documentation


class Model(object, metaclass=ABCMeta):
    """
    A parent class for all models of the integrated function in the BQ quadrature context. It is intended to be used
    by the subclasses of the `BQTransform` (i.e. Gaussian process and t-process quadrature moment transforms). The
    Model class ties together the kernel and the point-set used by the underlying quadrature rule. In modelling
    terms, the Model is composed of a kernel and point-set, that is, `Model` *has-a* `Kernel` and `points`.

    Assumptions
    -----------
      - The model of the integrand relies on a Kernel class, that is, it is either a GP or TP regression model.

    Attributes
    ----------
    Model._supported_points_ : list
        Each element of the list is an acronym of a point-set.
    Model._supported_kernels_ : list
        Each element of the list is an acronym of a kernel.
    kernel : Kernel
        Kernel used by the Model.
    points : numpy.ndarray
        Quadrature rule point-set.
    str_pts : string
    str_pts_par : string
        String representation of the kernel parameter values.
    emv : float
        Expected model variance.
    ivar : float
        Variance of the integral.
    dim_in : int
        Dimension of the point-set.
    num_pts : int
        Number of points.
    eye_d
    eye_n : numpy.ndarray
        Pre-allocated identity matrices to ease the computations.
    """

    _supported_points_ = ['sr', 'ut', 'gh']  # TODO: register fully-symmetric sets
    _supported_kernels_ = ['rbf']  # TODO: register RQ kernel

    def __init__(self, dim, kern_par, kernel, points, point_par=None):
        """
        Initialize model of the integrand with specified kernel and point set.

        Parameters
        ----------
        dim : int
            Dimension of the points (integration domain).
        kern_par : numpy.ndarray
            Kernel parameters in a vector.
        kernel : string
            String abbreviation for the kernel.
        points : string
            String abbreviation for the point-set.
        point_par : dict
            Any parameters for constructing desired point-set.
        """

        # init kernel and sigma-points
        self.kernel = Model.get_kernel(dim, kernel, kern_par)
        self.points = Model.get_points(dim, points, point_par)

        # save for printing
        self.str_pts = points
        self.str_pts_par = str(point_par)

        # may no longer be necessary now that jitter is in kernel
        self.dim_in, self.num_pts = self.points.shape
        self.eye_d, self.eye_n = np.eye(self.dim_in), np.eye(self.num_pts)

    def __str__(self):
        """
        Prettier string representation.

        Returns
        -------
        string
            String representation including short name of the point-set, the kernel and its parameter values.
        """
        return '{}\n{} {}'.format(self.kernel, self.str_pts, self.str_pts_par)

    @abstractmethod
    def predict(self, test_data, fcn_obs, par=None):
        """
        Model predictions based on test points and the kernel parameters.

        Notes
        -----
        This is an abstract method. Implementation needs to be provided by the subclass.

        Parameters
        ----------
        test_data : numpy.ndarray
            Test points where to generate data.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        par : numpy.ndarray
            Kernel parameters, default `par=None`.

        Returns
        -------
        (mean, var)
            Model predictive mean and variance at the test point locations.
        """
        pass

    @abstractmethod
    def exp_model_variance(self, fcn_obs, par=None):
        """
        Expected model variance given the function observations and the kernel parameters.

        Notes
        -----
        This is an abstract method. Implementation needs to be provided by the subclass and should be easily
        accomplished using the kernel expectation method from the `Kernel` class.

        Parameters
        ----------
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        par : numpy.ndarray
            Kernel parameters, default `par=None`.

        Returns
        -------
        float
            Expected model variance.
        """
        pass

    @abstractmethod
    def integral_variance(self, fcn_obs, par=None):
        """
        Integral variance given the function value observations and the kernel parameters.

        Notes
        -----
        This is an abstract method. Implementation needs to be provided by the subclass and should be easily
        accomplished using the kernel expectation method from the `Kernel` class.

        Parameters
        ----------
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        par : numpy.ndarray
            Kernel parameters, default `par=None`.

        Returns
        -------
        float
            Variance of the integral.
        """
        pass

    @abstractmethod
    def neg_log_marginal_likelihood(self, log_par, fcn_obs):
        """
        Negative logarithm of marginal likelihood of the model given the kernel parameters and the function
        observations.

        Notes
        -----
        Intends to be used as an objective function passed into the optimizer, thus it needs to subscribe to certain
        implementation conventions.

        Parameters
        ----------
        log_par : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
        float
            Negative log marginal likelihood.

        """
        pass

    def likelihood_reg_emv(self, log_par, fcn_obs):
        """
        Negative marginal log-likelihood with a expected model variance as regularizer.

        Parameters
        ----------
        log_par : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
            Sum of negative marginal log-likelihood and expected model variance.
        """
        # negative marginal log-likelihood w/ additional regularizing term
        # regularizing terms: integral variance, expected model variance or both, prior on par
        nlml, nlml_grad = self.neg_log_marginal_likelihood(log_par, fcn_obs)
        # NOTE: not entirely sure regularization is usefull, because the regularized ML-II seems to give very similar
        # results to ML-II; this regularizer tends to prefer longer lengthscales
        reg = self.exp_model_variance(fcn_obs, par=np.exp(log_par))
        return nlml + reg

    def likelihood_reg_ivar(self, log_par, fcn_obs):
        """
        Negative marginal log-likelihood with a integral variance as regularizer.

        Parameters
        ----------
        log_par : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
            Sum of negative marginal log-likelihood and integral variance.
        """
        # negative marginal log-likelihood w/ additional regularizing term
        nlml, nlml_grad = self.neg_log_marginal_likelihood(log_par, fcn_obs)
        reg = self.integral_variance(fcn_obs, par=np.exp(log_par))
        return nlml + reg

    def optimize(self, log_par_0, fcn_obs, crit='NLML', method='BFGS', **kwargs):
        """
        Find optimal values of kernel parameters by minimizing chosen criterion given the point-set and the function
        observations.

        Parameters
        ----------
        log_par_0 : numpy.ndarray
            Initial guess of the kernel log-parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        crit : string
            Objective function to use as a criterion for finding optimal setting of kernel parameters. Possible
            values are:
              - 'nlml' : negative marginal log-likelihood,
              - 'nlml+emv' : NLML with expected model variance as regularizer,
              - 'nlml+ivar' : NLML with integral variance as regularizer.
        method : string
            Optimization method for `scipy.optimize.minimize`, default method='BFGS'.
        **kwargs
            Keyword arguments for the `scipy.optimize.minimize`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Results of the optimization in a dict-like structure returned by `scipy.optimize.minimize`.

        Notes
        -----
        The criteria using expected model variance and integral variance as regularizers ('nlml+emv', 'nlml+ivar')
        are somewhat experimental. I did not operate under any sound theoretical justification when implementing
        those. Just curious to see what happens, thus might be removed in the future.

        See Also
        --------
        scipy.optimize.minimize
        """
        crit = crit.lower()
        if crit == 'nlml':
            obj_func = self.neg_log_marginal_likelihood
            jac = True
        elif crit == 'nlml+emv':
            obj_func = self.likelihood_reg_emv
            jac = False  # gradients not implemented for regularizers (solver uses approximations)
        elif crit == 'nlml+ivar':
            obj_func = self.likelihood_reg_ivar
            jac = False  # gradients not implemented for regularizers (solver uses approximations)
        else:
            raise ValueError('Unknown criterion {}.'.format(crit))
        return minimize(obj_func, log_par_0, fcn_obs, method=method, jac=jac, **kwargs)

    def plot_model(self, test_data, fcn_obs, par=None, fcn_true=None, in_dim=0):
        """
        Plot of predictive mean and variance of the fitted model of the integrand. Since we're plotting a function with
        multiple inputs and outputs, we need to specify which is to be plotted.

        Notes
        -----
        Not tested very much, likely to misbehave.

        Parameters
        ----------
        test_data : numpy.ndarray
            1D array of locations, where the function is to be evaluated for plotting.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.
        par : numpy.ndarray
            Kernel parameters, default `par=None`.
        fcn_true :
            True function values
        in_dim : int
            Index of the input dimension to plot.

        Returns
        -------

        """
        assert in_dim <= self.dim_in - 1

        fcn_obs = np.squeeze(fcn_obs)
        fcn_true = np.squeeze(fcn_true)

        # model predictive mean and variance
        mean, var = self.predict(test_data, fcn_obs, par=par)
        std = np.sqrt(var)
        test_data = np.squeeze(test_data[in_dim, :])

        # set plot title according to model
        fig_title = self.__class__.__name__ + ' model of the integrand'

        # plot training data, predictive mean and variance
        fig = plt.figure(fig_title)
        plt.fill_between(test_data, mean - 2 * std, mean + 2 * std, color='0.1', alpha=0.15)
        plt.plot(test_data, mean, color='k', lw=2)
        plt.plot(self.points[in_dim, :], fcn_obs, 'ko', ms=8)

        # true function values at test points if provided
        if fcn_true is not None:
            plt.plot(test_data, fcn_true, lw=2, ls='--', color='tomato')
        plt.show()

    @staticmethod
    def get_points(dim, points, point_par):
        """
        Construct desired point-set for integration. Calls methods of classical quadrature classes.

        Parameters
        ----------
        dim : int

        points : string
            String abbreviation for the point-set.
        point_par : dict
            Parameters for constructing desired point-set.

        Returns
        -------
        numpy.ndarray
            Point set in (D, N) array, where D is dimension and N number of points.

        Notes
        -----
        List of supported points is kept in ``_supported_points_`` class variable.
        """

        points = points.lower()

        # make sure points is supported
        if points not in Model._supported_points_:
            print('Points {} not supported. Supported points are {}.'.format(points, Model._supported_points_))
            return None
        if point_par is None:
            point_par = {}

        # create chosen points
        if points == 'sr':
            return SphericalRadial.unit_sigma_points(dim)
        elif points == 'ut':
            return Unscented.unit_sigma_points(dim, **point_par)
        elif points == 'gh':
            return GaussHermite.unit_sigma_points(dim, **point_par)

    @staticmethod
    def get_kernel(dim, kernel, par):
        """
        Initializes desired kernel.

        Parameters
        ----------
        dim : int
            Dimension of input (integration domain).
        kernel : string
            String abbreviation of the kernel.
        par : numpy.ndarray
            Parameters of the kernel.

        Returns
        -------
        : Kernel
            A subclass of Kernel.

        Notes
        -----
        List of supported kernels is kept in ``_supported_kernels_`` class variable.
        """

        kernel = kernel.lower()

        # make sure kernel is supported
        if kernel not in Model._supported_kernels_:
            print('Kernel {} not supported. Supported kernels are {}.'.format(kernel, Model._supported_kernels_))
            return None

        # initialize the chosen kernel
        if kernel == 'rbf':
            return RBF(dim, par)


class GaussianProcess(Model):  # consider renaming to GaussianProcessRegression/GPRegression, same for TP
    """
    Gaussian process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kern_par, kernel='rbf', points='ut', point_par=None):
        """
        Gaussian process regression model.

        Parameters
        ----------
        dim : int
            Number of input dimensions.
        kern_par : numpy.ndarray
            Kernel parameters in matrix.
        kernel : string
            Acronym of the covariance function of the Gaussian process model.
        points : string
            Acronym for the sigma-point set to use in BQ.
        point_par : dict
            Parameters of the sigma-point set.
        """

        super(GaussianProcess, self).__init__(dim, kern_par, kernel, points, point_par)

    def predict(self, test_data, fcn_obs, par=None):
        """
        Gaussian process predictions.

        Parameters
        ----------
        test_data : numpy.ndarray
            Test data, shape (D, M)
        fcn_obs : numpy.ndarray
            Observations of the integrand at sigma-points.
        par : numpy.ndarray
            Kernel parameters.

        Returns
        -------
        : tuple
            Predictive mean and variance in a tuple (mean, var).

        """

        par = self.kernel.get_parameters(par)

        iK = self.kernel.eval_inv_dot(par, self.points)
        kx = self.kernel.eval(par, test_data, self.points)
        kxx = self.kernel.eval(par, test_data, test_data, diag=True)

        # GP mean and predictive variance
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        return mean, var

    def exp_model_variance(self, fcn_obs=None, par=None):
        """

        Parameters
        ----------
        fcn_obs : numpy.ndarray
        par : numpy.ndarray

        Returns
        -------
        : float

        """

        par = self.kernel.get_parameters(par)
        Q = self.kernel.exp_x_kxkx(par, par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
        return self.kernel.scale.squeeze() ** 2 * (1 - np.trace(Q.dot(iK)))

    def integral_variance(self, fcn_obs, par=None):
        """

        Parameters
        ----------
        fcn_obs : numpy.ndarray
        par : numpy.ndarray

        Returns
        -------
        : float

        """

        par = self.kernel.get_parameters(par)
        q = self.kernel.exp_x_kx(par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
        kbar = self.kernel.exp_x_kxx(par)
        return kbar - q.T.dot(iK).dot(q)

    def neg_log_marginal_likelihood(self, log_par, fcn_obs):
        """
        Negative marginal log-likelihood of Gaussian process regression model.

        Parameters
        ----------
        log_par : numpy.ndarray
            Kernel log-parameters, shape (num_par, ).
        fcn_obs : numpy.ndarray
            Function values, shape (num_pts, dim_out).

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.

        Returns
        -------
        Negative log-likelihood and gradient for given parameter.

        """

        # convert from log-par to par
        par = np.exp(log_par)

        L = self.kernel.eval_chol(par, self.points)  # (N, N)
        K = L.dot(L.T)  # jitter included from eval_chol
        a = la.solve(K, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ji', fcn_obs.T, a)  # sum of diagonal of A.T.dot(A)
        a_out_a = np.einsum('i...j, ...jn', a, a.T)  # (N, N) sum over of outer products of columns of A

        # negative marginal log-likelihood
        nlml = 0.5 * y_dot_a + np.sum(np.log(np.diag(L))) + 0.5 * self.num_pts * np.log(2 * np.pi)
        # nlml = np.log(nlml)  # w/o this, unconstrained solver terminates w/ 'precision loss'

        # TODO: check the gradient with check_grad method
        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_par(par, self.points)  # (N, N, num_par)
        iK = la.solve(K, np.eye(self.num_pts))
        dnlml_dtheta = 0.5 * np.trace((iK - a_out_a).dot(dK_dTheta))  # (num_par, )
        return nlml, dnlml_dtheta


class GaussianProcessMO(Model):
    """
    Multi-output Gaussian process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim_in, dim_out, kern_par, kernel, points, point_par=None):
        """
        Multi-output Gaussian process regression model.

        Parameters
        ----------
        dim_in : int
            Number of input dimensions.
        dim_out : int
            Number of output dimensions.
        kern_par : numpy.ndarray
            Kernel parameters in matrix.
        kernel : string
            Acronym of the covariance function of the Gaussian process model.
        points : string
            Acronym for the sigma-point set to use in BQ.
        point_par : dict
            Parameters of the sigma-point set.
        """

        super(GaussianProcessMO, self).__init__(dim_in, kern_par, kernel, points, point_par)
        self.dim_out = dim_out

    def predict(self, test_data, fcn_obs, par=None):
        """
        Predictions of multi-output Gaussian process regression model.

        Parameters
        ----------
        test_data : numpy.ndarray
            Test data, shape (dim_in, num_test)
        fcn_obs : numpy.ndarray
            Observations of the integrand at sigma-points, shape (dim_out, num_pts)?
        par : numpy.ndarray
            Kernel parameters.

        Returns
        -------
        : tuple
            Predictive mean and variance in a tuple (mean, var).

        """
        pass

    def exp_model_variance(self, fcn_obs, par=None):
        par = self.kernel.get_parameters(par)

        emv = np.zeros((self.dim_out,))
        for i in range(self.dim_out):
            iK = self.kernel.eval_inv_dot(par[i, :], self.points, scaling=False)
            Q = self.kernel.exp_x_kxkx(par[i, :], par[i, :], self.points)
            emv[i] = self.kernel.scale[i] ** 2 * (1 - np.trace(Q.dot(iK)))
        return emv

    def integral_variance(self, fcn_obs, par=None):
        par = self.kernel.get_parameters(par)

        ivar = np.zeros((self.dim_out,))
        for i in range(self.dim_out):
            q = self.kernel.exp_x_kx(par[i, :], self.points)
            iK = self.kernel.eval_inv_dot(par[i, :], self.points, scaling=False)
            kbar = self.kernel.exp_x_kxx(par[i, :])
            ivar[i] = kbar - q.T.dot(iK).dot(q)
        return ivar

    def neg_log_marginal_likelihood(self, log_par, fcn_obs):
        """
        Negative marginal log-likelihood of a multi-output GP regression model.

        Parameters
        ----------
        log_par : numpy.ndarray
            Kernel log-parameters, shape (dim_out, num_par).
        fcn_obs : numpy.ndarray
            Function values, shape (num_pts, dim_out).

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.

        Returns
        -------
        Negative log-likelihood and gradient for given parameter.

        """
        pass


class StudentTProcess(Model):
    """
    Student t process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kern_par, kernel='rbf', points='ut', point_par=None, nu=3.0):
        """
        Student t process regression model.

        Parameters
        ----------
        dim : int
            Number of input dimensions.
        kern_par : numpy.ndarray
            Kernel parameters in matrix.
        kernel : string
            Acronym of the covariance function of the Gaussian process model.
        points : string
            Acronym for the sigma-point set to use in BQ.
        point_par : dict
            Parameters of the sigma-point set.
        """

        super(StudentTProcess, self).__init__(dim, kern_par, kernel, points, point_par)
        nu = 3.0 if nu < 2 else nu  # nu > 2
        self.nu = nu

    def predict(self, test_data, fcn_obs, par=None, nu=None):
        """
        Student t process predictions.

        Parameters
        ----------
        test_data : numpy.ndarray
            Test data, shape (D, M)
        fcn_obs : numpy.ndarray
            Observations of the integrand at sigma-points.
        par : numpy.ndarray
            Kernel parameters.
        nu : float
            Degrees of freedom.

        Returns
        -------
        : tuple
            Predictive mean and variance in a tuple (mean, var).

        """

        par = self.kernel.get_parameters(par)
        if nu is None:
            nu = self.nu

        iK = self.kernel.eval_inv_dot(par, self.points)
        kx = self.kernel.eval(par, test_data, self.points)
        kxx = self.kernel.eval(par, test_data, test_data, diag=True)
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        scale = (nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (nu - 2 + self.num_pts)
        return mean, scale * var

    def exp_model_variance(self, fcn_obs, par=None):
        """

        Parameters
        ----------
        fcn_obs
        par

        Returns
        -------

        """

        par = self.kernel.get_parameters(par)

        Q = self.kernel.exp_x_kxkx(par, par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
        fcn_obs = np.squeeze(fcn_obs)
        scale = (self.nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (self.nu - 2 + self.num_pts)
        return scale * self.kernel.scale.squeeze() ** 2 * (1 - np.trace(Q.dot(iK)))

    def integral_variance(self, fcn_obs, par=None):
        """

        Parameters
        ----------
        fcn_obs
        par

        Returns
        -------

        """

        par = self.kernel.get_parameters(par)

        kbar = self.kernel.exp_x_kxx(par)
        q = self.kernel.exp_x_kx(par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)

        fcn_obs = np.squeeze(fcn_obs)
        scale = (self.nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (self.nu - 2 + self.num_pts)
        return scale * (kbar - q.T.dot(iK).dot(q))

    def neg_log_marginal_likelihood(self, log_par, fcn_obs):
        """
        Negative marginal log-likelihood of Student t process regression model.

        Parameters
        ----------
        log_par : numpy.ndarray
            Kernel log-parameters, shape (num_par, ).
        fcn_obs : numpy.ndarray
            Function values, shape (num_pts, dim_out).

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.

        Returns
        -------
        Negative log-likelihood and gradient for given parameter.

        """

        # convert from log-par to par
        par = np.exp(log_par)

        L = self.kernel.eval_chol(par, self.points)  # (num_pts, num_pts)
        K = L.dot(L.T)  # jitter included from eval_chol
        a = la.solve(K, fcn_obs)  # (num_pts, dim_out)
        y_dot_a = np.einsum('ij, ji', fcn_obs.T, a)  # sum of diagonal of A.T.dot(A)
        a_out_a = np.einsum('i...j, ...jn', a, a.T)  # (num_pts, num_pts) sum over of outer products of columns of A

        # negative marginal log-likelihood
        from scipy.special import gamma
        half_logdet_K = np.sum(np.log(np.diag(L)))
        const = 0.5 * self.num_pts * np.log((self.nu - 2) * np.pi) + np.log(
            gamma(0.5 * self.nu + self.num_pts) / gamma(0.5 * self.nu))
        nlml = 0.5 * (self.nu + self.num_pts) * np.log(1 + y_dot_a) + half_logdet_K + const

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_par(par, self.points)  # (num_pts, num_pts, num_par)
        iK = la.solve(K, np.eye(self.num_pts))
        scale = (self.nu + self.num_pts) / (self.nu + y_dot_a - 2)
        dnlml_dtheta = 0.5 * np.trace((iK - scale * a_out_a).dot(dK_dTheta))  # (num_par, )
        return nlml, dnlml_dtheta
