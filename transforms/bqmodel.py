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
      - The regression models use single set of kernel parameters for all every output of the integrand.

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
    str_pts_hyp : string
        String representation of the kernel parameter values.
    emv : float
        Expected model variance.
    ivar : float
        Variance of the integral.
    d : int
        Dimension of the point-set.
    n : int
        Number of points.
    eye_d
    eye_n : numpy.ndarray
        Pre-allocated identity matrices to ease the computations.
    """
    _supported_points_ = ['sr', 'ut', 'gh']
    _supported_kernels_ = ['rbf']

    def __init__(self, dim, kernel, points, kern_hyp=None, point_hyp=None):
        """
        Initialize model of the integrand with specified kernel and point set.

        Parameters
        ----------
        dim : int
            Dimension of the points (integration domain).
        kernel : string
            String abbreviation for the kernel.
        points : string
            String abbreviation for the point-set.
        kern_hyp : numpy.ndarray
            Kernel parameters in a vector.
        point_hyp : numpy.ndarray
            Any parameters for constructing desired point-set.
        """

        # init kernel and sigma-points
        self.kernel = Model.get_kernel(dim, kernel, kern_hyp)
        self.points = Model.get_points(dim, points, point_hyp)

        # save for printing
        self.str_pts = points
        self.str_pts_hyp = str(point_hyp)

        # may no longer be necessary now that jitter is in kernel
        self.d, self.n = self.points.shape
        self.eye_d, self.eye_n = np.eye(self.d), np.eye(self.n)

    def __str__(self):
        """
        Prettier string representation.

        Returns
        -------
        string
            String representation including short name of the point-set, the kernel and its parameter values.
        """
        return '{}\n{} {}'.format(self.kernel, self.str_pts, self.str_pts_hyp)

    @abstractmethod
    def predict(self, test_data, fcn_obs, hyp=None):
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
        hyp : numpy.ndarray
            Kernel parameters, default `hyp=None`.

        Returns
        -------
        (mean, var)
            Model predictive mean and variance at the test point locations.
        """
        pass

    @abstractmethod
    def exp_model_variance(self, fcn_obs, hyp=None):
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
        hyp : numpy.ndarray
            Kernel parameters, default `hyp=None`.

        Returns
        -------
        float
            Expected model variance.
        """
        pass

    @abstractmethod
    def integral_variance(self, fcn_obs, hyp=None):
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
        hyp : numpy.ndarray
            Kernel parameters, default `hyp=None`.

        Returns
        -------
        float
            Variance of the integral.
        """
        pass

    @abstractmethod
    def neg_log_marginal_likelihood(self, log_hyp, fcn_obs):
        """
        Negative logarithm of marginal likelihood of the model given the kernel parameters and the function
        observations.

        Notes
        -----
        Intends to be used as an objective function passed into the optimizer, thus it needs to subscribe to certain
        implementation conventions.

        Parameters
        ----------
        log_hyp : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
        float
            Negative log marginal likelihood.

        """
        pass

    def likelihood_reg_emv(self, log_hyp, fcn_obs):
        """
        Negative marginal log-likelihood with a expected model variance as regularizer.

        Parameters
        ----------
        log_hyp : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
            Sum of negative marginal log-likelihood and expected model variance.
        """
        # negative marginal log-likelihood w/ additional regularizing term
        # regularizing terms: integral variance, expected model variance or both, prior on hypers
        nlml, nlml_grad = self.neg_log_marginal_likelihood(log_hyp, fcn_obs)
        # NOTE: not entirely sure regularization is usefull, because the regularized ML-II seems to give very similar
        # results to ML-II; this regularizer tends to prefer longer lengthscales
        reg = self.exp_model_variance(fcn_obs, hyp=np.exp(log_hyp))
        return nlml + reg

    def likelihood_reg_ivar(self, log_hyp, fcn_obs):
        """
        Negative marginal log-likelihood with a integral variance as regularizer.

        Parameters
        ----------
        log_hyp : numpy.ndarray
            Logarithm of the kernel parameters.
        fcn_obs : numpy.ndarray
            Observed function values at the point-set locations.

        Returns
        -------
            Sum of negative marginal log-likelihood and integral variance.
        """
        # negative marginal log-likelihood w/ additional regularizing term
        nlml, nlml_grad = self.neg_log_marginal_likelihood(log_hyp, fcn_obs)
        reg = self.integral_variance(fcn_obs, hyp=np.exp(log_hyp))
        return nlml + reg

    def optimize(self, log_hyp0, fcn_obs, crit='NLML', method='BFGS', **kwargs):
        """
        Find optimal values of kernel parameters by minimizing chosen criterion given the point-set and the function
        observations.

        Parameters
        ----------
        log_hyp0 : numpy.ndarray
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
        return minimize(obj_func, log_hyp0, fcn_obs, method=method, jac=jac, **kwargs)

    def plot_model(self, test_data, fcn_obs, hyp=None, fcn_true=None, in_dim=0):
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
        hyp : numpy.ndarray
            Kernel parameters, default `hyp=None`.
        fcn_true :
            True function values
        in_dim : int
            Index of the input dimension to plot.

        Returns
        -------

        """
        assert in_dim <= self.d - 1

        fcn_obs = np.squeeze(fcn_obs)
        fcn_true = np.squeeze(fcn_true)
        # model predictive mean and variance
        mean, var = self.predict(test_data, fcn_obs, hyp=hyp)
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
    def get_points(dim, points, point_hyp):
        """
        Construct desired point-set for integration. Calls methods of classical quadrature classes.

        Parameters
        ----------
        dim : int

        points : string
            String abbreviation for the point-set.
        point_hyp : numpy.ndarray
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
        if point_hyp is None:
            point_hyp = {}
        # create chosen points
        if points == 'sr':
            return SphericalRadial.unit_sigma_points(dim)
        elif points == 'ut':
            return Unscented.unit_sigma_points(dim, **point_hyp)
        elif points == 'gh':
            return GaussHermite.unit_sigma_points(dim, **point_hyp)

    @staticmethod
    def get_kernel(dim, kernel, hypers):
        """
        Initializes desired kernel.

        Parameters
        ----------
        dim : int
            Dimension of input (integration domain).
        kernel : string
            String abbreviation of the kernel.
        hypers : numpy.ndarray
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
            return RBF(dim, 1, hypers)
        elif kernel == 'affine':
            return Affine(dim, hypers)


class GaussianProcess(Model):  # consider renaming to GaussianProcessRegression/GPRegression, same for TP
    """
    Gaussian process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        """
        Gaussian process model

        Parameters
        ----------
        dim : int
            Number of input dimensions
        kernel : string
            Acronym of the covariance function of the Gaussian process model.
        points : string
            Acronym for the sigma-point set to use in BQ.
        kern_hyp : dict
            Kernel parameters in dictionary.
        point_hyp : dict
            Parameters of the sigma-point set.
        """
        super(GaussianProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)

    def predict(self, test_data, fcn_obs, hyp=None):
        """

        Parameters
        ----------
        test_data
        fcn_obs
        hyp

        Returns
        -------

        """

        iK = self.kernel.eval_inv_dot(hyp, self.points)
        kx = self.kernel.eval(hyp, test_data, self.points)
        kxx = self.kernel.eval(hyp, test_data, test_data, diag=True)

        # GP mean and predictive variance
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        return mean, var

    def exp_model_variance(self, fcn_obs, hyp=None):
        """

        Parameters
        ----------
        fcn_obs
        hyp

        Returns
        -------

        """

        hyp = self.kernel.get_hyperparameters(hyp)
        Q = self.kernel.exp_x_kxkx(self.points, hyp, hyp)
        iK = self.kernel.eval_inv_dot(hyp, self.points, scaling=False)
        return self.kernel.scale ** 2 * (1 - np.trace(Q.dot(iK)))

    def integral_variance(self, fcn_obs, hyp=None):
        """

        Parameters
        ----------
        fcn_obs
        hyp

        Returns
        -------

        """

        hyp = self.kernel.get_hyperparameters(hyp)
        q = self.kernel.exp_x_kx(self.points, hyp)
        iK = self.kernel.eval_inv_dot(hyp, self.points, scaling=False)
        kbar = self.kernel.exp_x_kxx(hyp)
        return kbar - q.T.dot(iK).dot(q)

    def neg_log_marginal_likelihood(self, log_hyp, fcn_obs):
        """

        Parameters
        ----------
        log_hyp
        fcn_obs

        Returns
        -------

        """
        # marginal log-likelihood of GP, uses log-hypers for optimization reasons
        # N - # points, E - # function outputs
        # fcn_obs (N, E), hypers (num_hyp, )

        # convert from log-hypers to hypers
        hypers = np.exp(log_hyp)

        L = self.kernel.eval_chol(self.points, hyp=hypers)  # (N, N)
        K = L.dot(L.T)  # jitter included from eval_chol
        a = la.solve(K, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ji', fcn_obs.T, a)  # sum of diagonal of A.T.dot(A)
        a_out_a = np.einsum('i...j, ...jn', a, a.T)  # (N, N) sum over of outer products of columns of A
        # negative marginal log-likelihood
        nlml = 0.5 * y_dot_a + np.sum(np.log(np.diag(L))) + 0.5 * self.n * np.log(2 * np.pi)
        # nlml = np.log(nlml)  # w/o this, unconstrained solver terminates w/ 'precision loss'
        # TODO: check the gradient with check_grad method
        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_hyp(self.points, hypers)  # (N, N, num_hyp)
        iK = la.solve(K, np.eye(self.n))
        dnlml_dtheta = 0.5 * np.trace((iK - a_out_a).dot(dK_dTheta))  # (num_hyp, )
        return nlml, dnlml_dtheta


class GaussianProcessMO(Model):
    """
    Multi-output Gaussian process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None):
        pass

    def predict(self, test_data, fcn_obs, hyp=None):
        pass

    def exp_model_variance(self, fcn_obs, hyp=None):
        hyp = self.kernel.get_hyperparameters(hyp)

        emv = np.zeros((self.e, ))
        for i in range(self.e):
            iK = self.kernel.eval_inv_dot(hyp[i, :], self.points, scaling=False)
            Q = self.kernel.exp_x_kxkx(self.points, hyp[i, :], hyp[i, :])
            emv[i] = self.kernel.scale[i] ** 2 * (1 - np.trace(Q.dot(iK)))
        return emv

    def integral_variance(self, fcn_obs, hyp=None):
        hyp = self.kernel.get_hyperparameters(hyp)

        ivar = np.zeros((self.e, ))
        for i in range(self.e):
            q = self.kernel.exp_x_kx(self.points, hyp[i, :])
            iK = self.kernel.eval_inv_dot(hyp[i, :], self.points, scaling=False)
            kbar = self.kernel.exp_x_kxx(hyp[i, :])
            ivar[i] = kbar - q.T.dot(iK).dot(q)
        return ivar

    def neg_log_marginal_likelihood(self, log_hyp, fcn_obs):
        pass


class StudentTProcess(Model):
    """
    Student t process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kernel='rbf', points='ut', kern_hyp=None, point_hyp=None, nu=None):
        """

        Parameters
        ----------
        dim
        kernel
        points
        kern_hyp
        point_hyp
        nu
        """
        super(StudentTProcess, self).__init__(dim, kernel, points, kern_hyp, point_hyp)
        assert nu > 2, 'Degrees of freedom (nu) must be > 2.'
        nu = 3.0 if nu is None else nu
        self.nu = nu

    def predict(self, test_data, fcn_obs, hyp=None, nu=None):
        """

        Parameters
        ----------
        test_data
        fcn_obs
        hyp
        nu

        Returns
        -------

        """
        if nu is None:
            nu = self.nu
        iK = self.kernel.eval_inv_dot(self.points)
        kx = self.kernel.eval(test_data, self.points)
        kxx = self.kernel.eval(test_data, test_data, diag=True)
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,mi->i', kx, iK, kx.T))
        scale = (nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (nu - 2 + self.n)
        return mean, scale * var

    def exp_model_variance(self, fcn_obs, hyp=None):
        """

        Parameters
        ----------
        fcn_obs
        hyp

        Returns
        -------

        """

        hyp = self.kernel.get_hyperparameters(hyp)

        Q = self.kernel.exp_x_kxkx(self.points, hyp, hyp)
        iK = self.kernel.eval_inv_dot(hyp, self.points, scaling=False)
        fcn_obs = np.squeeze(fcn_obs)
        scale = (self.nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (self.nu - 2 + self.n)
        return scale * self.kernel.scale ** 2 * (1 - np.trace(Q.dot(iK)))

    def integral_variance(self, fcn_obs, hyp=None):
        """

        Parameters
        ----------
        fcn_obs
        hyp

        Returns
        -------

        """

        hyp = self.kernel.get_hyperparameters(hyp)

        kbar = self.kernel.exp_x_kxx(hyp)
        q = self.kernel.exp_x_kx(self.points, hyp)
        iK = self.kernel.eval_inv_dot(hyp, self.points, scaling=False)

        fcn_obs = np.squeeze(fcn_obs)
        scale = (self.nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (self.nu - 2 + self.n)
        return scale * (kbar - q.T.dot(iK).dot(q))

    def neg_log_marginal_likelihood(self, log_hyp, fcn_obs):
        """

        Parameters
        ----------
        log_hyp
        fcn_obs

        Returns
        -------

        """
        # marginal log-likelihood of TP, uses log-hypers for optimization reasons
        # N - # points, E - # function outputs
        # fcn_obs (N, E), hypers (num_hyp, )

        # convert from log-hypers to hypers
        hypers = np.exp(log_hyp)

        L = self.kernel.eval_chol(self.points, hyp=hypers)  # (N, N)
        K = L.dot(L.T)  # jitter included from eval_chol
        a = la.solve(K, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ji', fcn_obs.T, a)  # sum of diagonal of A.T.dot(A)
        a_out_a = np.einsum('i...j, ...jn', a, a.T)  # (N, N) sum over of outer products of columns of A

        # negative marginal log-likelihood
        from scipy.special import gamma
        half_logdet_K = np.sum(np.log(np.diag(L)))
        const = 0.5 * self.n * np.log((self.nu - 2) * np.pi) + np.log(
            gamma(0.5 * self.nu + self.n) / gamma(0.5 * self.nu))
        nlml = 0.5 * (self.nu + self.n) * np.log(1 + y_dot_a) + half_logdet_K + const

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_hyp(self.points, hypers)  # (N, N, num_hyp)
        iK = la.solve(K, np.eye(self.n))
        scale = (self.nu + self.n) / (self.nu + y_dot_a - 2)
        dnlml_dtheta = 0.5 * np.trace((iK - scale * a_out_a).dot(dK_dTheta))  # (num_hyp, )
        return nlml, dnlml_dtheta
