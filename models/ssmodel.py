import numpy as np
from numpy import newaxis as na
# TODO: Abstract Base Classes to enforce the requirements of the base class on the derived classes.


class StateSpaceModel(object):

    xD = None  # state dimension
    zD = None  # measurement dimension
    qD = None  # state noise dimension
    rD = None  # measurement noise dimension
    q_additive = None  # True = state noise is additive, False = non-additive
    r_additive = None
    # lists the keyword arguments currently required by the StateSpaceModel class
    _required_kwargs_ = 'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov', 'q_factor'

    def __init__(self, **kwargs):
        self.pars = kwargs
        self.zero_q = np.zeros((self.qD))
        self.zero_r = np.zeros((self.rD))

    def dyn_fcn(self, x, q, pars):
        """ System dynamics.

        Abstract method for the system dynamics.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like
            Parameters of the system dynamics

        Returns
        -------
        1-D numpy.ndarray of shape (self.xD,)
            system state in the next time step
        """
        raise NotImplementedError

    def meas_fcn(self, x, r, pars):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            system state
        r : 1-D array_like of shape (self.rD,)
            measurement noise
        pars : 1-D array_like
            parameters of the measurement model

        Returns
        -------
        1-D numpy.ndarray of shape (self.zD,)
            measurement of the state
        """
        raise NotImplementedError

    def par_fcn(self, time):
        """Parameter function of the system dynamics and measurement model.

        Abstract method for the parameter function of the whole state-space model. The implementation should ensure
        that the system dynamics parameters come before the measurement model parameters in the returned vector of
        parameters.

        Parameters
        ----------
        time : int
            Discrete time step

        Returns
        -------
        1-D numpy.ndarray of shape (self.pD,)
            Vector of parameters at a given time.
        """
        raise NotImplementedError

    def dyn_fcn_dx(self, x, q, pars):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        q : 1-D array_like of shape (self.qD,)
            System noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the system dynamics, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.qD)
        """
        raise NotImplementedError

    def meas_fcn_dx(self, x, r, pars):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : 1-D array_like of shape (self.xD,)
            System state
        r : 1-D array_like of shape (self.qD,)
            Measurement noise
        pars : 1-D array_like of shape (self.pD,)
            System parameter

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (self.xD, self.xD)
                * non-additive: (self.xD, self.xD + self.rD)
        """
        raise NotImplementedError

    def dyn_eval(self, xq, pars, dx=False):
        """Evaluation of the system dynamics according to noise additivity.

        Parameters
        ----------
        xq : 1-D array_like
            Augmented system state
        pars : 1-D array_like
            System dynamics parameters
        dx : bool
            * ``True``: Evaluates derivatives (Jacobian) of the system dynamics
            * ``False``: Evaluates system dynamics
        Returns
        -------
            Evaluated system dynamics or evaluated Jacobian of the system dynamics.
        """
        if self.q_additive:
            assert len(xq) == self.xD
            if dx:  # TODO: put in zero vector, not int
                out = (self.dyn_fcn_dx(xq, self.zero_q, pars).T.flatten())
            else:
                out = self.dyn_fcn(xq, self.zero_q, pars)
        else:
            assert len(xq) == self.xD + self.qD
            x, q = xq[:self.xD], xq[-self.qD:]
            if dx:
                out = (self.dyn_fcn_dx(x, q, pars).T.flatten())
            else:
                out = self.dyn_fcn(x, q, pars)
        return out

    def meas_eval(self, xr, pars, dx=False):
        """Evaluation of the system dynamics according to noise additivity.

        Parameters
        ----------
        xr : 1-D array_like
            Augmented system state
        pars : 1-D array_like
            Measurement model parameters
        dx : bool
            * ``True``: Evaluates derivatives (Jacobian) of the measurement model
            * ``False``: Evaluates measurement model
        Returns
        -------
            Evaluated measurement model or evaluated Jacobian of the measurement model.
        """
        if self.r_additive:
            assert len(xr) == self.xD
            if dx:
                out = (self.meas_fcn_dx(xr, self.zero_r, pars).T.flatten())
            else:
                out = self.meas_fcn(xr, self.zero_r, pars)
        else:
            assert len(xr) == self.xD + self.rD
            x, r = xr[:self.xD], xr[-self.rD:]
            if dx:
                out = (self.meas_fcn_dx(x, r, pars).T.flatten())
            else:
                out = self.meas_fcn(x, r, pars)
        return out

    def check_jacobians(self, h=1e-8):
        """Checks implemented Jacobians.

        Checks that both Jacobians are correctly implemented using numerical approximations.

        Parameters
        ----------
        h : float
            step size in derivative approximations

        Returns
        -------
            Prints the errors and user decides whether they're acceptable.
        """
        nq = self.xD if self.q_additive else self.xD + self.qD
        nr = self.xD if self.r_additive else self.xD + self.rD
        xq, xr = np.random.rand(nq), np.random.rand(nr)
        hq_diag, hr_diag = np.diag(h * np.ones(nq)), np.diag(h * np.ones(nr))
        assert hq_diag.shape == (nq, nq) and hr_diag.shape == (nr, nr)
        xqph, xqmh = xq[:, na] + hq_diag, xq[:, na] - hq_diag
        xrph, xrmh = xr[:, na] + hr_diag, xr[:, na] - hr_diag
        par = self.par_fcn(1.0)
        fph = np.zeros((self.xD, nq))
        hph = np.zeros((self.zD, nr))
        fmh, hmh = fph.copy(), hph.copy()
        for i in range(nq):
            fph[:, i] = self.dyn_eval(xqph[:, i], par)
            fmh[:, i] = self.dyn_eval(xqmh[:, i], par)
        for i in range(nr):
            hph[:, i] = self.meas_eval(xrph[:, i], par)
            hmh[:, i] = self.meas_eval(xrmh[:, i], par)
        jac_fx = (2 * h) ** -1 * (fph - fmh)
        jac_hx = (2 * h) ** -1 * (hph - hmh)
        print "Errors in Jacobians\n{}\n{}".format(np.abs(jac_fx - self.dyn_eval(xq, par, dx=True)),
                                                   np.abs(jac_hx - self.meas_eval(xr, par, dx=True)))

    def simulate(self, steps, mc_sims=1):
        """State-space model simulation.

        SSM simulation starting from initial conditions for a given number of time steps

        Parameters
        ----------
        steps : int
            Number of time steps in state trajectory
        mc_sims : int
            Number of trajectories to simulate (the initial state is drawn randomly)

        Returns
        -------
        tuple
            Tuple (x, z) where both element are of type numpy.ndarray and where:

                * x : 3-D array of shape (self.xD, steps, mc_sims) containing the true system state trajectory
                * z : 3-D array of shape (self.zD, steps, mc_sims) containing simulated measurements of the system state
        """

        x0_mean, x0_cov, q_mean, q_cov, r_mean, r_cov = self.get_pars(
                'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov'
        )
        x = np.zeros((self.xD, steps, mc_sims))
        z = np.zeros((self.zD, steps, mc_sims))
        q = np.random.multivariate_normal(q_mean, q_cov, size=(mc_sims, steps)).T
        r = np.random.multivariate_normal(r_mean, r_cov, size=(mc_sims, steps)).T
        x0 = np.random.multivariate_normal(x0_mean, x0_cov, size=mc_sims).T  # (D, mc_sims)
        x[:, 0, :] = x0  # store initial states at k=0
        for imc in xrange(mc_sims):
            for k in xrange(1, steps):
                theta = self.par_fcn(k - 1)
                x[:, k, imc] = self.dyn_fcn(x[:, k-1, imc], q[:, k-1, imc], theta)
                z[:, k, imc] = self.meas_fcn(x[:, k, imc], r[:, k, imc], theta)
        return x, z

    def set_pars(self, key, value):
        self.pars[key] = value

    def get_pars(self, *keys):
        values = []
        for k in keys:
            values.append(self.pars.get(k))
        return values
