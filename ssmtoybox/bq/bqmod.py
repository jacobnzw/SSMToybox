from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.special import factorial, factorial2

from .bqkern import RBFGauss, RQ, RBFStudent
from ssmtoybox.mtran import SphericalRadialTransform, UnscentedTransform, GaussHermiteTransform, \
    FullySymmetricStudentTransform
from ssmtoybox.utils import vandermonde, n_sum_k


class Model(object, metaclass=ABCMeta):
    """
    Base class for all models of the integrated function in the BQ quadrature context. It is intended to be used
    by the subclasses of the `BQTransform` (i.e. Gaussian process and t-process quadrature moment transforms). The
    Model class ties together the kernel and the point-set used by the underlying quadrature rule. In modelling
    terms, the Model is composed of a kernel and point-set, that is, `Model` *has-a* `Kernel` and `points`.

    Parameters
    ----------
    dim : int
        Dimension of the points (integration domain).

    kern_par : ndarray
        Kernel parameters in a vector.

    kern_str : str
        String abbreviation for the kernel.

    point_str : str
        String abbreviation for the point-set.

    point_par : dict
        Any parameters for constructing desired point-set.

    estimate_par : bool
        Set to `True` if you wanna re-compute kernel expectations based on changing parameters, like in
        `MarginalInference`.

    Attributes
    ----------
    Model._supported_points_ : list
        Each element of the list is an acronym of a point-set.

    Model._supported_kernels_ : list
        Each element of the list is an acronym of a kernel.

    kernel : Kernel
        Kernel used by the Model.

    points : ndarray
        Quadrature rule point-set.

    str_pts : str
    str_pts_par : str
        String representation of the kernel parameter values.

    emv : float
        Expected model variance.

    ivar : float
        Variance of the integral.

    dim_in : int
        Dimension of the point-set.

    num_pts : int
        Number of points.

    eye_d : ndarray
    eye_n : ndarray
        Pre-allocated identity matrices to ease the computations.

    Notes
    -----
    The model of the integrand relies on a Kernel class, that is, it is either a GP or TP regression model.
    """

    _supported_points_ = ['sr', 'ut', 'gh', 'fs']
    _supported_kernels_ = ['rbf', 'rq', 'rbf-student']

    def __init__(self, dim, kern_par, kern_str, point_str, point_par, estimate_par):
        # init kernel and sigma-points
        self.kernel = Model.get_kernel(dim, kern_str, kern_par)
        self.points = Model.get_points(dim, point_str, point_par)

        # turn on/off re-computation of kernel expectations
        self.estimate_par = estimate_par

        # save for printing
        self.str_pts = point_str
        self.str_pts_par = str(point_par)

        # may no longer be necessary now that jitter is in kernel
        self.dim_in, self.num_pts = self.points.shape
        self.eye_d, self.eye_n = np.eye(self.dim_in), np.eye(self.num_pts)

        # init variables for passing kernel expectations and kernel matrix inverse
        self.q, self.Q, self.R, self.iK = None, None, None, None

        # expected model variance and integral model variance
        self.model_var = None
        self.integral_var = None

    def __str__(self):
        """
        Prettier string representation.

        Returns
        -------
        : str
            String representation including short name of the point-set, the kernel and its parameter values.
        """
        return '{}\n{} {}'.format(self.kernel, self.str_pts, self.str_pts_par)

    @abstractmethod
    def predict(self, test_data, fcn_obs, par=None):
        """
        Model predictions based on test points and the kernel parameters.

        Parameters
        ----------
        test_data : ndarray
            Test points where to generate data.

        fcn_obs : ndarray
            Observed function values at the point-set locations.

        par : ndarray
            Kernel parameters, default `par=None`.

        Returns
        -------
        mean : ndarray
            Model predictive mean at the test point locations.

        var : ndarray
            Model predictive variance at the test point locations.

        Notes
        -----
        This is an abstract method. Implementation needs to be provided by the subclass.
        """
        pass

    @abstractmethod
    def bq_weights(self, par, *args):
        """
        Weights of the Bayesian quadrature.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        Returns
        -------
        wm : ndarray
            Weights for computation of the transformed mean.

        Wc : ndarray
            Weights for computation of the transformed covariance.

        Wcc : ndarray
            Weights for computation of the transformed cross-covariance.

        emvar : ndarray
            Expected model variance.

        ivar : ndarray
            Integral variance.
        """
        pass

    @abstractmethod
    def exp_model_variance(self, par, *args):
        """
        Expected model variance :math:`\mathbb{E}_x[\mathbb{V}_f[f(x)]]`.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        Returns
        -------
        : float
            Expected model variance.
        """
        pass

    @abstractmethod
    def integral_variance(self, par, *args):
        """
        Integral variance :math:`\mathbb{V}_f[\mathbb{E}_x[f(x)]]`.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        Returns
        -------
        : float
            Variance of the integral.
        """
        pass

    @abstractmethod
    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative logarithm of marginal likelihood of the model given the kernel parameters and the function
        observations.

        Parameters
        ----------
        log_par : ndarray
            Logarithm of the kernel parameters.

        fcn_obs : ndarray
            Observed function values at the inputs supplied in `x_obs`.

        x_obs : ndarray
            Function inputs.

        jitter : ndarray
            Regularization term for kernel matrix inversion.

        Returns
        -------
        float
            Negative log marginal likelihood.

        Notes
        -----
        Intends to be used as an objective function passed into the optimizer, thus it needs to subscribe to certain
        implementation conventions.
        """
        pass

    def optimize(self, log_par_0, fcn_obs, x_obs, method='BFGS', **kwargs):
        """Optimize kernel parameters.

        Find optimal values of kernel parameters by minimizing chosen criterion given the point-set and the function
        observations.

        Parameters
        ----------
        log_par_0 : ndarray
            Initial guess of the kernel log-parameters.

        fcn_obs : ndarray
            Observed function values at the point-set locations.

        x_obs : ndarray
            Function inputs.

        method : str
            Optimization method for `scipy.optimize.minimize`, default method='BFGS'.

        **kwargs
            Keyword arguments for the `scipy.optimize.minimize`.

        Returns
        -------
        : scipy.optimize.OptimizeResult
            Results of the optimization in a dict-like structure returned by `scipy.optimize.minimize`.

        See Also
        --------
        scipy.optimize.minimize
        """
        obj_func = self.neg_log_marginal_likelihood
        jac = True
        jitter = 1e-8 * np.eye(x_obs.shape[1])
        return minimize(obj_func, log_par_0, args=(fcn_obs, x_obs, jitter), method=method, jac=jac, **kwargs)

    def plot_model(self, test_data, fcn_obs, par=None, fcn_true=None, in_dim=0):
        """
        Plot of predictive mean and variance of the fitted model of the integrand. Since we're plotting a function with
        multiple inputs and outputs, we need to specify which is to be plotted.

        Parameters
        ----------
        test_data : ndarray
            1D array of locations, where the function is to be evaluated for plotting.

        fcn_obs : ndarray
            Observed function values at the point-set locations.

        par : ndarray
            Kernel parameters, default `par=None`.

        fcn_true :
            True function values.

        in_dim : int
            Index of the input dimension to plot.

        Returns
        -------

        Notes
        -----
        Not tested very much, likely to misbehave.
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
        : ndarray
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
            return SphericalRadialTransform.unit_sigma_points(dim)
        elif points == 'ut':
            return UnscentedTransform.unit_sigma_points(dim, **point_par)
        elif points == 'gh':
            return GaussHermiteTransform.unit_sigma_points(dim, **point_par)
        elif points == 'fs':
            return FullySymmetricStudentTransform.unit_sigma_points(dim, **point_par)

    @staticmethod
    def get_kernel(dim, kernel, par):
        """
        Initializes desired kernel.

        Parameters
        ----------
        dim : int
            Dimension of input (integration domain).

        kernel : str
            String abbreviation of the kernel.

        par : ndarray
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
            return RBFGauss(dim, par)
        elif kernel == 'rbf-student':
            return RBFStudent(dim, par)
        elif kernel == 'rq':
            return RQ(dim, par)


class GaussianProcessModel(Model):
    """
    Gaussian process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kern_par, kern_str, point_str, point_par=None, estimate_par=False):
        """
        Gaussian process regression model.

        Parameters
        ----------
        estimate_par
        dim : int
            Number of input dimensions.

        kern_par : ndarray
            Kernel parameters in matrix.

        kern_str : str
            Acronym of the covariance function of the Gaussian process model.

        point_str : str
            Acronym for the sigma-point set to use in BQ.

        point_par : dict
            Parameters of the sigma-point set.
        """

        super(GaussianProcessModel, self).__init__(dim, kern_par, kern_str, point_str, point_par, estimate_par)

    def predict(self, test_data, fcn_obs, x_obs=None, par=None):
        """
        Gaussian process predictions.

        Parameters
        ----------
        test_data : ndarray
            Test data, shape (D, M).

        fcn_obs : ndarray
            Observations of the integrand at sigma-points.

        x_obs : ndarray
            Training inputs.

        par : ndarray
            Kernel parameters.

        Returns
        -------
        mean : ndarray
            Predictive mean.

        var : ndarray
            Predictive variance.
        """

        if x_obs is None:
            x_obs = self.points

        par = self.kernel.get_parameters(par)

        iK = self.kernel.eval_inv_dot(par, x_obs)
        kx = self.kernel.eval(par, test_data, x_obs)
        kxx = self.kernel.eval(par, test_data, test_data, diag=True)

        # GP mean and predictive variance
        mean = np.squeeze(kx.dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,ni->i', kx, iK, kx.T))
        return mean, var

    def bq_weights(self, par, *args):
        par = self.kernel.get_parameters(par)
        x = self.points

        # inverse kernel matrix
        iK = self.kernel.eval_inv_dot(par, x, scaling=False)

        # Kernel expectations
        q = self.kernel.exp_x_kx(par, x)
        Q = self.kernel.exp_x_kxkx(par, par, x)
        R = self.kernel.exp_x_xkx(par, x)

        # BQ weights in terms of kernel expectations
        w_m = q.dot(iK)
        w_c = iK.dot(Q).dot(iK)
        w_cc = R.dot(iK)

        # save the kernel expectations for later
        self.q, self.Q, self.iK = q, Q, iK
        # expected model variance
        self.model_var = self.kernel.exp_x_kxx(par) * (1 - np.trace(Q.dot(iK)))
        # integral variance
        self.integral_var = self.kernel.exp_xy_kxy(par) - q.T.dot(iK).dot(q)

        # covariance weights should be symmetric
        if not np.array_equal(w_c, w_c.T):
            w_c = 0.5 * (w_c + w_c.T)

        return w_m, w_c, w_cc, self.model_var, self.integral_var

    def exp_model_variance(self, par, *args):
        iK = self.kernel.eval_inv_dot(par, self.points)
        Q = self.kernel.exp_x_kxkx(par, par, self.points)
        return self.kernel.exp_x_kxx(par) * (1 - np.trace(Q.dot(iK)))

    def integral_variance(self, par, *args):
        par = self.kernel.get_parameters(par)  # if par None returns default kernel parameters
        q = self.kernel.exp_x_kx(par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
        kbar = self.kernel.exp_xy_kxy(par)
        return kbar - q.T.dot(iK).dot(q)

    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative marginal log-likelihood of single-output Gaussian process regression model.

        The likelihood is given by

        .. math::
        \[
        -\\log p(Y \\mid X, \\theta) = -\\sum_{e=1}^{\\mathrm{dim_out}} \\log p(y_e \\mid X, \\theta)
        \]

        where :math:`y_e` is e-th column of :math:`Y`. We have the same parameters :math:`\theta` for all outputs,
        which is more limiting than the multi-output case. For single-output dimension the expression is equivalent to
        negative marginal log-likelihood.

        Parameters
        ----------
        log_par : (num_par, ) ndarray
            Kernel log-parameters.

        fcn_obs : (num_pts, dim_out) ndarray
            Function values.

        x_obs : ndarray
            Function inputs.

        jitter : ndarray
            Regularization term for kernel matrix inversion.

        Returns
        -------
        : float
            Negative log-likelihood and gradient for given parameter.

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.
        """

        # convert from log-par to par
        par = np.exp(log_par)
        num_data, num_out = fcn_obs.shape

        K = self.kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ji', fcn_obs.T, a)  # sum of diagonal of A.T.dot(A)
        a_out_a = np.einsum('i...j, ...jn', a, a.T)  # (N, N) sum over of outer products of columns of A

        # negative total NLML
        nlml = num_out * np.sum(np.log(np.diag(L[0]))) + 0.5 * (y_dot_a + num_out * num_data * np.log(2 * np.pi))

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_par(par, x_obs)  # (N, N, num_hyp)
        iKdK = la.cho_solve(L, dK_dTheta)

        # gradient of total NLML
        dnlml_dtheta = 0.5 * np.trace((num_out * iKdK - a_out_a.dot(dK_dTheta)))  # (num_par, )

        return nlml, dnlml_dtheta


class BayesSardModel(Model):
    """
    Gaussian process model for Bayes-Sard quadrature. The model has multivariate polynomial prior mean.

    Parameters
    ----------
    dim : int
        Dimension of the points (integration domain).

    kern_par : ndarray
        Kernel parameters in a vector.

    tdeg : int, optional
        Total degree of the multivariate polynomial GP prior mean.

    points : str, optional
        String abbreviation for the point-set.

    point_par : dict, optional
        Any parameters for constructing desired point-set.
    """

    def __init__(self, dim, kern_par, multi_ind=2, point_str='ut', point_par=None, estimate_par=False):
        super(BayesSardModel, self).__init__(dim, kern_par, 'rbf', point_str, point_par, estimate_par)

        if type(multi_ind) is int:
            # multi-index: monomials of total degree <= multi_ind
            self.mulind = []
            for td in range(multi_ind + 1):
                self.mulind.append(n_sum_k(dim, td))
            self.mulind = np.hstack(self.mulind)
        elif type(multi_ind) is np.ndarray:
            self.mulind = multi_ind
        else:
            raise ValueError('Multi-index error: multi-index has to be either int or ndarray')

    def _exp_x_px(self, multi_ind):
        """
        Compute expectation \\mathbb{E}[p(x)^T]_{q} for all :math:`q`. The expectation is equal to

        .. math::
             \\prod_{d=1}^D (\\alpha_d^q - 1)!!

        when :math:`\\alpha^q_d` is even :math:`\\forall q`. Otherwise the expectation is zero.

        Parameters
        ----------
        multi_ind : (D, Q) ndarray
            Matrix of multi-indices. Each column is a multi-index :math:`\\alpha^q \\in \\mathbb{N}_0^D` defining one
            of the Q multivariate polynomial basis functions.

        Returns
        -------
        : (Q, ) ndarray
            Vector of expectations.
        """
        dim, num_basis = multi_ind.shape
        alpha = multi_ind - 1
        result = np.zeros((num_basis, ))
        for q in range(num_basis):
            all_even = np.all(multi_ind[:, q] % 2 == 0)
            if all_even:
                result[q] = np.prod([factorial2(alpha[d, q], exact=True) for d in range(dim)])
        return result

    def _exp_x_xpx(self, multi_ind):
        """
        Compute expectation \\mathbb{E}[xp(x)^T]_{eq} for all :math:`e` and :math:`q`. The expectation is equal to

        .. math::
             \\alpha^q_e\\prod_{d \neq e} (\\alpha^q_d - 1)!!

        when :math:`\\alpha^q_e + 1` is even and :math:`\\alpha^q_d, \\forall d \neq e` are even.
        Otherwise the expectation is zero.

        Parameters
        ----------
        multi_ind : (D, Q) ndarray
            Matrix of multi-indices. Each column is a multi-index :math:`\\alpha^q \\in \\mathbb{N}_0^D` defining one
            of the Q multivariate polynomial basis functions.

        Returns
        -------
        : (D, Q) ndarray
            Matrix of expectations.
        """
        dim, num_bases = multi_ind.shape
        d_ind = np.arange(dim)
        result = np.zeros(multi_ind.shape)
        for d in range(dim):
            for q in range(num_bases):
                # all remaining multi-indices even? # i.e. none are odd?
                alpha_min_d = multi_ind[d_ind != d, q]
                all_even = np.all(alpha_min_d % 2 == 0)
                if (multi_ind[d, q] + 1) % 2 == 0 and all_even:
                    amd_fact2 = [factorial2(amd - 1, exact=True) for amd in alpha_min_d]
                    result[d, q] = multi_ind[d, q]*np.prod(amd_fact2)
                else:
                    result[d, q] = 0
        return result

    def _exp_x_pxpx(self, multi_ind):
        """
        Compute expectation \\mathbb{E}[p(x)p(x)^T]_{rq} for all :math:`r` and :math:`q`. The expectation is equal to

        .. math::
             \\prod_{d = 1}^D (\\alpha^q_d + \\alpha^r_d - 1)!!

        when :math:`\\forall d,\\quad \\alpha^q_d + \\alpha^r_d` are even (where :math:`r` and :math:`q` are fixed).
        Otherwise the expectation is zero.

        Parameters
        ----------
        multi_ind : (D, Q) ndarray
            Matrix of multi-indices. Each column is a multi-index :math:`\\alpha^q \\in \\mathbb{N}_0^D` defining one
            of the Q multivariate polynomial basis functions.

        Returns
        -------
        : (Q, Q) ndarray
            Matrix of expectations.
        """
        dim, num_bases = multi_ind.shape
        result = np.zeros((num_bases, num_bases))
        for r in range(num_bases):
            for q in range(num_bases):
                all_even = np.all((multi_ind[:, r] + multi_ind[:, q]) % 2 == 0)
                if all_even:
                    apa_fact2 = [factorial2(multi_ind[d, r] + multi_ind[d, q] - 1, exact=True) for d in range(dim)]
                    result[r, q] = np.prod(apa_fact2)
                else:
                    result[r, q] = 0
        return result

    def _exp_x_kxpx(self, par, multi_ind, x):
        """
        Compute expectation :math:`\\mathbb{E}[k(x)p(x)^T]_{nq}`. For given :math:`n` and :math:`q`, the expectation is
        given by

        .. math::
            \\prod_{d=1}^D\left[ a_{ijd} b_{ijd} \right]

        where

        .. math::
            a_{ijd} = \\ell_d(1+\ell^2_d)^{-(1+\\alpha_{dj})/2} \\exp\left(-\\frac{x_{dj}^2}{2(1+\ell_d^2)}\right)

        .. math::
            b_{ijd} = \\sum_{m=0}^{\left\lfloor \\alpha_{dj}/2 \right\rfloor}
            \\frac{\\alpha_{dj}!}{2^m m! (\\alpha_{dj} - 2m)!}
            \\ell_d^{2m}\left(\\frac{x_{di}}{\\sqrt{1+\\ell^2_d}}\right)^{\\alpha_{dj}-2m}

        Parameters
        ----------
        par : (dim, ) ndarray
            Kernel parameters.

        multi_ind : (D, Q) ndarray
            Matrix of multi-indices. Each column is a multi-index :math:`\\alpha^q \\in \\mathbb{N}_0^D` defining one
            of the Q multivariate polynomial basis functions.

        x : (dim, N) ndarray
            Data points.

        Returns
        -------
        : (N, Q) ndarray
            Matrix of expectations.
        """
        dim, num_bases = multi_ind.shape
        num_pts = x.shape[1]
        scale, sqrt_inv_lam = self.kernel._unpack_parameters(par)
        ell = np.diag(sqrt_inv_lam) ** -2
        result = np.zeros((num_pts, num_bases))
        dim_zeros = np.zeros((dim, ))

        fact = lambda num: factorial(num, exact=True)
        for n in range(num_pts):
            for q in range(num_bases):

                # compute factors in the product
                temp = dim_zeros.copy()
                for d in range(dim):
                    alpha = multi_ind[d, q]
                    # exponential part
                    a = ell[d]*(1 + ell[d]**2)**(-(1+alpha)/2) * np.exp(-x[d, n]**2 / (2*(1 + ell[d]**2)))

                    # binomial part
                    b = 0
                    for m in range(int(np.floor(alpha/2))+1):
                        part_1 = (fact(alpha) / ((2**m) * fact(m) * fact(alpha - 2*m)))
                        part_2 = (ell[d]**(2*m)) * ((x[d, n]/np.sqrt(1+ell[d]**2))**(alpha - 2*m))
                        b += part_1 * part_2
                    temp[d] = a * b

                # final result
                result[n, q] = np.prod(temp)

        return result

    def _mc_exp_x_kxpx(self, par, multind, x):
        # approximate expectations using cumulative moving average MC
        def cma_mc(new_samples, old_avg, old_avg_size, axis=0):
            b_size = new_samples.shape[axis]
            return (new_samples.sum(axis=axis) + old_avg_size * old_avg) / (old_avg_size + b_size)

        dim = x.shape[0]
        batch_size = 100000
        num_iter = 10
        kxpx_mc = 0
        mean, cov = np.zeros((dim,)), np.eye(dim)
        for i in range(num_iter):
            # sample from standard Gaussian
            x_samples = np.random.multivariate_normal(mean, cov, size=batch_size).T
            p = vandermonde(multind, x_samples)  # (N, Q)
            k = self.kernel.eval(par, x_samples, x, scaling=False)  # (N, M)
            kxpx_mc = cma_mc(k[..., None] * p[:, None, :], kxpx_mc, i * batch_size, axis=0)
        return kxpx_mc

    def _mc_exp_x_cov(self, par, multind, x):
        # approximate expectations using cumulative moving average MC
        def cma_mc(new_samples, old_avg, old_avg_size, axis=0):
            b_size = new_samples.shape[axis]
            return (new_samples.sum(axis=axis) + old_avg_size * old_avg) / (old_avg_size + b_size)

        dim = x.shape[0]
        batch_size = 100000
        num_iter = 10
        cov_mc = 0
        mean, cov = np.zeros((dim,)), np.eye(dim)
        V = vandermonde(multind, x)
        ViK = V.T.dot(self.kernel.eval_inv_dot(par, x))
        for i in range(num_iter):
            # sample from standard Gaussian
            x_samples = np.random.multivariate_normal(mean, cov, size=batch_size).T
            p = vandermonde(multind, x_samples)  # (N, Q)
            k = self.kernel.eval(par, x_samples, x, scaling=False)  # (N, M)
            b = k.dot(ViK.T) - p
            cov_mc = cma_mc(b[..., None] * b[:, None, :], cov_mc, i * batch_size, axis=0)
        return cov_mc

    def predict(self, test_data, fcn_obs, x_obs=None, par=None, mulind=None):
        """
        Gaussian process predictions with polynomial prior mean.

        Parameters
        ----------
        test_data : ndarray
            Test data, shape (D, M).

        fcn_obs : ndarray
            Observations of the integrand at sigma-points.

        x_obs : ndarray, optional
            Training inputs.

        par : (1, num_par) ndarray, optional
            Kernel parameters.

        mulind : (dim, num_basis) ndarray, optional
            Matrix where each column is multi-index specifying a multivariate monomial.

        Returns
        -------
        mean : ndarray
            Predictive mean.

        var : ndarray
            Predictive variance.
        """
        if x_obs is None:
            x_obs = self.points
        # if multi-index unspecified, revert to default degree multivariate polynomial
        if mulind is None:
            mulind = self.mulind
        num_basis = mulind.shape[1]
        par = self.kernel.get_parameters(par)

        iK = self.kernel.eval_inv_dot(par, x_obs)  # (num_train, num_train)
        kx = self.kernel.eval(par, test_data, x_obs)  # (num_test, num_train)
        kxx = self.kernel.eval(par, test_data, test_data, diag=True)  # (num_test, num_test)

        V = vandermonde(mulind, x_obs)  # (num_train, num_basis)
        Z = V.T.dot(iK)  # (num_basis, num_train)
        iViKV = la.cho_solve(la.cho_factor(Z.dot(V)), np.eye(num_basis))  # (num_basis, num_basis)
        A = iViKV.dot(V.T)  # (num_basis, num_train)
        vx = vandermonde(mulind, test_data)  # (num_test, num_basis)
        b = Z.dot(kx.T) - vx.T  # (num_basis, num_test)

        # GP mean and predictive variance
        mean = np.squeeze((kx - b.T.dot(A)).dot(iK).dot(fcn_obs.T))
        var = np.squeeze(kxx - np.einsum('im,mn,ni->i', kx, iK, kx.T) + np.einsum('im,mn,ni->i', b.T, iViKV, b))
        return mean, var

    def bq_weights(self, par, multi_ind=None):
        """
        Weights for the Bayes-Sard quadrature.

        Parameters
        ----------
        par : (1, num_par) ndarray
            Kernel parameters.

        multi_ind : (dim, num_basis) ndarray
            Matrix, where each column is a multi-index defining a multivariate monomial.

        Returns
        -------
        wm : ndarray
            Weights for computation of the transformed mean.

        Wc : ndarray
            Weights for computation of the transformed covariance.

        Wcc : ndarray
            Weights for computation of the transformed cross-covariance.

        emvar : ndarray
            Expected model variance.

        ivar : ndarray
            Integral variance.
        """
        if multi_ind is None:
            multi_ind = self.mulind
        par = self.kernel.get_parameters(par)
        x = self.points
        num_basis = multi_ind.shape[1]

        # dimension of the monomials must be equal to dimension of the points
        if multi_ind.shape[0] != self.dim_in:
            raise ValueError('Dimension mismatch {:d} != {:d}. Dimension of monomials must be equal to the dimension'
                             ' of the sigma-points.'.format(multi_ind.shape[0], self.dim_in))

        # inverse kernel matrix and Vandermonde matrix
        iK = self.kernel.eval_inv_dot(par, x, scaling=False)
        V = vandermonde(multi_ind, x)
        iViKV = la.cho_solve(la.cho_factor(V.T.dot(iK).dot(V) + 1e-8 * np.eye(num_basis)), np.eye(num_basis))
        # expectations of multivariate polynomials
        px = self._exp_x_px(multi_ind)
        xpx = self._exp_x_xpx(multi_ind)
        pxpx = self._exp_x_pxpx(multi_ind)
        kxpx = self._exp_x_kxpx(par, multi_ind, x)
        # kernel expectations
        q = self.kernel.exp_x_kx(par, x)
        kxy = self.kernel.exp_xy_kxy(par)

        # if the number of basis functions is same as the number of points, we assume the points are pi-unisolvent
        # and use a special-case implementation of quadrature weights
        if num_basis == self.num_pts:
            # inverse Vandermonde matrix
            iV = la.solve(V, np.eye(num_basis))

            # BSQ weights for pi-unisolvent points
            w_m = iV.T.dot(px)
            w_c = iV.T.dot(pxpx).dot(iV)
            w_cc = xpx.dot(iV)

            # expected model variance and integral variance
            kscale2 = self.kernel.scale.squeeze()**2
            # pxpx.dot(iViKV) == w_c.dot(K)
            self.model_var = kscale2 * (1 - np.trace(kxpx.T.dot(iV.T) + kxpx.dot(iV) - pxpx.dot(iViKV)))
            self.integral_var = kxy - q.T.dot(iV.T).dot(px) - px.T.dot(iV).dot(q) + px.T.dot(iViKV).dot(px)

        elif num_basis < self.num_pts:
            # additional kernel expectations
            Q = self.kernel.exp_x_kxkx(par, par, x)
            R = self.kernel.exp_x_xkx(par, x)

            Z = V.T.dot(iK)
            A = V.dot(iViKV)
            b = Z.dot(q) - px
            B = Z.dot(Q).dot(Z.T) + pxpx - Z.dot(kxpx) - kxpx.T.dot(Z.T)
            D = R.dot(Z.T) - xpx

            # BSQ weights: general case
            w_m = iK.dot(q - A.dot(b))
            w_c = iK.dot(Q - A.dot(B).dot(A.T)).dot(iK)
            w_cc = (R - D.dot(A.T)).dot(iK)

            # expected model variance and integral variance
            kscale2 = self.kernel.scale.squeeze() ** 2
            self.model_var = kscale2 * (1 - np.trace(Q.dot(iK)) + np.trace(B.dot(iViKV)))
            self.integral_var = kxy - q.T.dot(iK).dot(q) + b.T.dot(iViKV).dot(b)

        else:
            raise ValueError('Number of basis functions needs to be lower than or equal to the number of points.'
                             'You supplied {:d} basis functions and {:d} points.'.format(num_basis, self.num_pts))

        # covariance weights should be symmetric
        if not np.array_equal(w_c, w_c.T):
            w_c = 0.5 * (w_c + w_c.T)

        return w_m, w_c, w_cc, self.model_var, self.integral_var

    def exp_model_variance(self, par, mulind=None):
        """
        Expected model variance for the Bayes-Sard GP model of the integrand.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        mulind : ndarray, optional
            Multi-index.

        Returns
        -------
        : float
            Expected model variance.
        """
        if mulind is None:
            mulind = self.mulind
        par = self.kernel.get_parameters(par)

        pxpx = self._exp_x_pxpx(mulind)
        kxpx = self._exp_x_kxpx(par, mulind, self.points)
        kxkx = self.kernel.exp_x_kxkx(par, par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
        V = vandermonde(mulind, self.points)
        iViKV = la.cho_solve(la.cho_factor(V.T.dot(iK).dot(V)), np.eye(mulind.shape[1]))
        Z = V.T.dot(iK)
        B = Z.dot(kxkx).dot(Z.T) + pxpx - Z.dot(kxpx) - kxpx.T.dot(Z.T)
        kscale = self.kernel.scale.squeeze() ** 2
        return kscale * (1 - np.trace(kxkx.dot(iK)) + np.trace(B.dot(iViKV)))

    def integral_variance(self, par, mulind=None):
        """
        Variance of the integral.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        mulind : ndarray, optional
            Multi-index.

        Returns
        -------
        : float

        """
        if mulind is None:
            mulind = self.mulind
        par = self.kernel.get_parameters(par)

        q = self.kernel.exp_x_kx(par, self.points)
        iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
        kbar = self.kernel.exp_xy_kxy(par)
        V = vandermonde(mulind, self.points)
        px = self._exp_x_px(mulind)
        b = V.T.dot(iK).dot(q) - px
        iViKV = la.cho_solve(la.cho_factor(V.T.dot(iK).dot(V)), np.eye(mulind.shape[1]))
        return kbar - q.T.dot(iK).dot(q) + b.T.dot(iViKV).dot(b)

    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        pass


class StudentTProcessModel(GaussianProcessModel):
    """
    Student t process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim, kern_par, kern_str, point_str, point_par=None, estimate_par=False, nu=4.0):
        """
        Student's t-process regression model.

        Parameters
        ----------
        estimate_par
        dim : int
            Number of input dimensions.

        kern_par : ndarray
            Kernel parameters in matrix.

        kern_str : str
            Acronym of the covariance function of the Gaussian process model.

        point_str : str
            Acronym for the sigma-point set to use in BQ.

        point_par : dict
            Parameters of the sigma-point set.

        """

        super(StudentTProcessModel, self).__init__(dim, kern_par, kern_str, point_str, point_par, estimate_par)
        nu = 3.0 if nu < 2 else nu  # nu > 2
        self.nu = nu

    def predict(self, test_data, fcn_obs, x_obs=None, par=None, nu=None):
        """
        Student's t-process predictions.

        Parameters
        ----------
        test_data : (D, M) ndarray
            Test data.

        fcn_obs : ndarray
            Observations of the integrand at sigma-points.

        x_obs : ndarray
            Training inputs.

        par : ndarray
            Kernel parameters.

        nu : float
            Degrees of freedom.

        Returns
        -------
        mean : ndarray
            Predictive mean.

        var : ndarray
            Predictive variance.

        """

        par = self.kernel.get_parameters(par)
        if nu is None:
            nu = self.nu
        if x_obs is None:
            x_obs = self.points

        mean, var = super(StudentTProcessModel, self).predict(test_data, fcn_obs, x_obs, par)
        iK = self.kernel.eval_inv_dot(par, self.points)
        scale = (nu - 2 + fcn_obs.T.dot(iK).dot(fcn_obs)) / (nu - 2 + self.num_pts)
        return mean, scale * var

    def exp_model_variance(self, par, *args):
        """
        Expected model variance.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        args[0] : ndarray
            Function observations.

        Returns
        -------
        : float
            Expected model variance.
        """

        fcn_obs = np.squeeze(args[0])
        if self.estimate_par:
            # re-compute EMV if kernel parameters are being estimated
            iK = self.kernel.exp_x_kxkx(par, par, self.points)
            scale = (self.nu - 2 + fcn_obs.dot(iK).dot(fcn_obs.T)) / (self.nu - 2 + self.num_pts)
            gp_emv = super(StudentTProcessModel, self).exp_model_variance(par)
        else:
            # otherwise use pre-computed values (based on unit sigma-points and kernel parameters given at init)
            scale = (self.nu - 2 + fcn_obs.dot(self.iK).dot(fcn_obs.T)) / (self.nu - 2 + self.num_pts)
            gp_emv = self.model_var
        return scale * gp_emv

    def integral_variance(self, par, *args):
        """
        Variance of the integral.

        Parameters
        ----------
        par : ndarray
            Kernel parameters.

        args[0] : ndarray
            Function evaluations.

        Returns
        -------
        : float

        """

        fcn_obs = np.squeeze(args[0])
        par = self.kernel.get_parameters(par)
        if self.estimate_par:
            iK = self.kernel.eval_inv_dot(par, self.points, scaling=False)
            scale = (self.nu - 2 + fcn_obs.dot(iK).dot(fcn_obs.T)) / (self.nu - 2 + self.num_pts)
            gp_ivar = super(StudentTProcessModel, self).integral_var(par)
        else:
            scale = (self.nu - 2 + fcn_obs.dot(self.iK).dot(fcn_obs.T)) / (self.nu - 2 + self.num_pts)
            gp_ivar = self.integral_var
        return scale * gp_ivar

    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative marginal log-likelihood of Student's t-process regression model.

        Parameters
        ----------
        log_par : (num_par, ) ndarray
            Kernel log-parameters.

        fcn_obs : (num_pts, dim_out) ndarray
            Function values.

        x_obs : ndarray
            Function inputs.

        jitter : ndarray
            Regularization term for kernel matrix inversion.

        Returns
        -------
        : float
            Negative log-likelihood and gradient for given parameter.

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.
        """

        # convert from log-par to par
        par = np.exp(log_par)
        num_data, num_out = fcn_obs.shape
        nu = self.nu

        K = self.kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, E)
        y_dot_a = np.einsum('ij, ij -> j', fcn_obs, a)  # sum of diagonal of A.T.dot(A)

        # negative marginal log-likelihood
        from scipy.special import gamma
        half_logdet_K = np.sum(np.log(np.diag(L[0])))
        const = (num_data/2) * np.log((nu-2)*np.pi) - np.log(gamma((nu+num_data)/2)) + np.log(gamma(nu/2))
        log_sum = 0.5*(self.nu + num_data) * np.log(1 + y_dot_a/(self.nu - 2)).sum()
        nlml = log_sum + num_out*(half_logdet_K + const)

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_par(par, x_obs)  # (N, N, num_par)

        # gradient
        iKdK = la.cho_solve(L, dK_dTheta)
        scale = (self.nu + num_data) / (self.nu + y_dot_a - 2)
        a_out_a = np.einsum('j, i...j, ...jn', scale, a, a.T)  # (N, N) weighted sum of outer products of columns of A
        dnlml_dtheta = 0.5 * np.trace((num_out * iKdK - a_out_a.dot(dK_dTheta)))  # (num_par, )

        return nlml, dnlml_dtheta


class MultiOutputModel(Model):

    def __init__(self, dim_in, dim_out, kern_par, kern_str, point_str, point_par=None, estimate_par=False):
        super(MultiOutputModel, self).__init__(dim_in, kern_par, kern_str, point_str, point_par, estimate_par)
        self.dim_out = dim_out

    def bq_weights(self, par):
        """
        Weights of the Bayesian quadrature with multi-output process model.

        Parameters
        ----------
        par : (dim_out, num_par) ndarray
            Kernel parameters in a matrix, where e-th row contains parameters for e-th output.

        Returns
        -------
        wm : (num_pts, dim_out) ndarray
            Multi-output GP quadrature weights for the mean.

        wc : (num_pts, num_pts, dim_out, dim_out) ndarray
            Multi-output GP quadrature weights for the mean.

        wcc : (dim_in, num_pts, dim_out) ndarray
            Multi-output GP quadrature weights for the cross-covariance.
        """

        # if kern_par=None return parameters stored in Kernel
        par = self.kernel.get_parameters(par)

        # retrieve sigma-points from Model
        x = self.points
        d, e, n = self.dim_in, self.dim_out, self.num_pts

        # Kernel expectations
        q = np.zeros((n, e))
        Q = np.zeros((n, n, e, e))
        R = np.zeros((d, n, e))
        iK = np.zeros((n, n, e))
        w_c = np.zeros((n, n, e, e))
        for i in range(e):
            q[:, i] = self.kernel.exp_x_kx(par[i, :], x)
            R[..., i] = self.kernel.exp_x_xkx(par[i, :], x)
            iK[..., i] = self.kernel.eval_inv_dot(par[i, :], x, scaling=False)
            for j in range(i + 1):
                Q[..., i, j] = self.kernel.exp_x_kxkx(par[i, :], par[j, :], x)
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

    def optimize(self, log_par_0, fcn_obs, x_obs, method='BFGS', **kwargs):
        """
        Find optimal values of kernel parameters by minimizing negative marginal log-likelihood.

        Parameters
        ----------
        log_par_0 : ndarray
            Initial guess of the kernel log-parameters.

        fcn_obs : ndarray
            Observed function values at the point-set locations.

        x_obs : ndarray
            Function inputs.

        crit : str
            Objective function to use as a criterion for finding optimal setting of kernel parameters. Possible
            values are:
              - 'nlml' : negative marginal log-likelihood,
              - 'nlml+emv' : NLML with expected model variance as regularizer,
              - 'nlml+ivar' : NLML with integral variance as regularizer.

        method : str
            Optimization method for `scipy.optimize.minimize`, default method='BFGS'.

        **kwargs
            Keyword arguments for the `scipy.optimize.minimize`.

        Returns
        -------
        : scipy.optimize.OptimizeResult
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

        obj_func = self.neg_log_marginal_likelihood
        jitter = 1e-8 * np.eye(x_obs.shape[1])
        results = list()
        for d in range(self.dim_out):
            r = minimize(obj_func, log_par_0[d, :], args=(fcn_obs[d, :, None], x_obs, jitter),
                         method=method, jac=True, **kwargs)
            results.append(r)

        # extract optimized parameters and arrange in 2D array
        par = np.vstack([r.x for r in results])

        return par, results

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
    def exp_model_variance(self, fcn_obs):
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
    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative logarithm of marginal likelihood of the model given the kernel parameters and the function
        observations.

        Parameters
        ----------
        log_par : numpy.ndarray
            Logarithm of the kernel parameters.

        fcn_obs : numpy.ndarray
            Observed function values at the inputs supplied in `x_obs`.

        x_obs : numpy.ndarray
            Function inputs.

        jitter : numpy.ndarray
            Regularization term for kernel matrix inversion.

        Returns
        -------
        float
            Negative log marginal likelihood.

        Notes
        -----
        Intends to be used as an objective function passed into the optimizer, thus it needs to subscribe to certain
        implementation conventions.
        """
        pass


class GaussianProcessMO(MultiOutputModel):  # TODO: Multiple inheritance could be used here
    """
    Multi-output Gaussian process regression model of the integrand in the Bayesian quadrature.
    """

    def __init__(self, dim_in, dim_out, kern_par, kern_str, point_str, point_par=None):
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

        kern_str : string
            Acronym of the covariance function of the Gaussian process model.

        point_str : string
            Acronym for the sigma-point set to use in BQ.

        point_par : dict
            Parameters of the sigma-point set.
        """

        super(GaussianProcessMO, self).__init__(dim_in, dim_out, kern_par, kern_str, point_str, point_par)

    def predict(self, test_data, fcn_obs, par=None):
        """
        Predictions of multi-output Gaussian process regression model.

        Parameters
        ----------
        test_data : numpy.ndarray
            Test data, shape (dim_in, num_test).

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

    def exp_model_variance(self, fcn_obs):

        emv = np.zeros((self.dim_out, ))
        for i in range(self.dim_out):
            emv[i] = self.kernel.scale[i] ** 2 * (1 - np.trace(self.Q[..., i, i].dot(self.iK[..., i])))
        return emv

    def integral_variance(self, fcn_obs, par=None):
        par = self.kernel.get_parameters(par)

        ivar = np.zeros((self.dim_out,))
        for i in range(self.dim_out):
            q = self.kernel.exp_x_kx(par[i, :], self.points)
            iK = self.kernel.eval_inv_dot(par[i, :], self.points, scaling=False)
            kbar = self.kernel.exp_xy_kxy(par[i, :])
            ivar[i] = kbar - q.T.dot(iK).dot(q)
        return ivar

    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative marginal log-likelihood of multi-output Gaussian process regression model.

        The likelihood is given by

        .. math::
        \[
        -\log p(Y \mid X, \Theta) = -\sum_{e=1}^{\mathrm{dim_out}} \log p(y_e \mid X, \theta_e)
        \]

        where :math:`y_e` is e-th column of :math:`Y` and :math:`\theta_e` is e-th column of :math:`\Theta`. The
        multi-output model uses one set of kenrel parameters for every output, thus having greater flexibility than the
        single-output GP models.the same parameters :math:`\theta` for all outputs, which is more limiting than the
        multi-output case. For single-output dimension the expression is equivalent to negative marginal log-likelihood.
        This function implements only one term in the sum above, because the outputs are assumed independent given
        the inputs and thus we can run the optimizer for each output independently.

        Parameters
        ----------
        log_par : numpy.ndarray
            Kernel log-parameters, shape (num_par, ).

        fcn_obs : numpy.ndarray
            Function values, shape (num_pts, dim_out).

        x_obs : numpy.ndarray
            Function inputs, shape ().

        jitter : numpy.ndarray
            Regularization term for kernel matrix inversion.

        Returns
        -------
        Negative log-likelihood and gradient for given parameter.

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.
        """

        # convert from log-par to par
        par = np.exp(log_par)
        num_data = x_obs.shape[1]

        K = self.kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, )
        y_dot_a = fcn_obs.T.dot(a)
        a_out_a = np.outer(a, a.T)  # (N, N)

        # negative marginal log-likelihood
        nlml = np.sum(np.log(np.diag(L[0]))) + 0.5 * (y_dot_a + num_data * np.log(2 * np.pi))

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_par(par, x_obs)  # (N, N, num_par)
        iKdK = la.cho_solve(L, dK_dTheta)
        dnlml_dtheta = 0.5 * np.trace((iKdK - a_out_a.dot(dK_dTheta)))  # (num_par, )

        return nlml, dnlml_dtheta


class StudentTProcessMO(MultiOutputModel):

    def __init__(self, dim_in, dim_out, kern_par, kern_str, point_str, point_par=None, nu=3.0):
        """
        Multi-output Student's t-process regression model.

        Parameters
        ----------
        dim_in : int
            Number of input dimensions.

        dim_out : int
            Number of output dimensions.

        kern_par : ndarray
            Kernel parameters in matrix.

        kern_str : str
            Acronym of the covariance function of the Gaussian process model.

        point_str : str
            Acronym for the sigma-point set to use in BQ.

        point_par : dict
            Parameters of the sigma-point set.
        """

        super(StudentTProcessMO, self).__init__(dim_in, dim_out, kern_par, kern_str, point_str, point_par)
        self.nu = nu

    def predict(self, test_data, fcn_obs, par=None):
        pass

    def exp_model_variance(self, fcn_obs):
        """

        Parameters
        ----------
        fcn_obs

        Returns
        -------

        """

        # fcn_obs = np.squeeze(fcn_obs)
        emv = np.zeros((self.dim_out, ))
        # NOTE: einsum could be used here
        for d in range(self.dim_out):
            scale = self.nu - 2 + fcn_obs[d, :].dot(self.iK[..., d]).dot(fcn_obs[d, :].T)
            scale /= (self.nu - 2 + self.num_pts)
            emv[d] = scale * (1 - np.trace(self.Q[..., d, d].dot(self.iK[..., d])))

        return self.kernel.scale.squeeze() ** 2 * emv

    def integral_variance(self, fcn_obs, par=None):
        pass

    def neg_log_marginal_likelihood(self, log_par, fcn_obs, x_obs, jitter):
        """
        Negative marginal log-likelihood of Student t process regression model.

        Parameters
        ----------
        log_par : ndarray
            Kernel log-parameters, shape (num_par, ).

        fcn_obs : ndarray
            Function values, shape (num_pts, dim_out).

        x_obs : ndarray
            Function inputs, shape ().

        jitter : ndarray
            Regularization term for kernel matrix inversion.

        Returns
        -------
        value : float
            Negative log-likelihood.
        grad : ndarray
            Negative log-likelihood gradient.

        Notes
        -----
        Used as an objective function by the `Model.optimize()` to find an estimate of the kernel parameters.
        """

        # convert from log-par to par
        par = np.exp(log_par)
        num_data = x_obs.shape[1]

        K = self.kernel.eval(par, x_obs) + jitter  # (N, N)
        L = la.cho_factor(K)  # jitter included from eval
        a = la.cho_solve(L, fcn_obs)  # (N, )
        y_dot_a = fcn_obs.T.dot(a)
        a_out_a = np.outer(a, a.T)  # (N, N)

        # negative marginal log-likelihood
        from scipy.special import gamma
        half_logdet_K = np.sum(np.log(np.diag(L)))
        const = 0.5 * num_data * np.log((self.nu - 2) * np.pi) + np.log(
            gamma(0.5 * self.nu + num_data) / gamma(0.5 * self.nu))
        nlml = 0.5 * (self.nu + num_data) * np.log(1 + y_dot_a) + half_logdet_K + const

        # negative marginal log-likelihood derivatives w.r.t. hyper-parameters
        dK_dTheta = self.kernel.der_par(par, x_obs)  # (N, N, num_par)
        iKdK = la.cho_solve(L, dK_dTheta)
        scale = (self.nu + num_data) / (self.nu + y_dot_a - 2)
        dnlml_dtheta = 0.5 * np.trace((iKdK - scale * a_out_a.dot(dK_dTheta)))  # (num_par, )

        return nlml, dnlml_dtheta