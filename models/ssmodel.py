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
    _required_kwargs_ = 'x0_mean', 'x0_cov', 'q_mean', 'q_cov', 'r_mean', 'r_cov'

    def __init__(self, **kwargs):
        self.pars = kwargs

    def dyn_fcn(self, x, q, pars):
        """
        Function of the system dynamics.

        :param x: 1-D array_like, of shape (self.xD,)
            system state
        :param q: 1-D array_like, of shape (self.qD,)
            system noise
        :param pars: 1-D array_like
            system parameter
        :return: ndarray, of shape (self.xD,)
            system state in the next time step
        """
        raise NotImplementedError

    def meas_fcn(self, x, r, pars):
        """
        Function of the measurement model.

        :param x: 1-D array_like, of shape (self.xD,)
            system state
        :param r: 1-D array_like, of shape (self.rD,)
            measurement noise
        :param pars: 1-D array_like
            system parameter
        :return: 1-D ndarray, of shape (self.zD,)
            measurement of the state
        """
        raise NotImplementedError

    def par_fcn(self, time):
        """
        Parameter function of the system dynamics and measurement model.

        :param time: time step
        :return: 1-D ndarray, of shape (self.pD,)
            parameter value at a given time, first dimensions are for system parameters,
            later for the measurement model parameters
        """
        raise NotImplementedError

    def dyn_fcn_dx(self, x, q, pars):
        """
        Jacobian of the system dynamics.

        :param x: 1-D array_like, of shape (self.xD,)
            system state
        :param q: 1-D array_like, of shape (self.qD,)
            system noise
        :param pars: 1-D array_like, of shape (self.pD,)
            system parameter
        :return: 2-D ndarray, of shape (self.xD, in_dim), where in = self.xD (add) or self.xD+self.qD (non-add)
            Jacobian matrix of the system dynamics, where the second dimension dependes on the noise additivity.
        """
        raise NotImplementedError

    def meas_fcn_dx(self, x, r, pars):
        """
        Jacobian of the measurement function.

        :param x: 1-D array_like, of shape (self.xD,)
            system state
        :param r: 1-D array_like, of shape (self.qD,)
            measurement noise
        :param pars: 1-D array_like, of shape (self.pD,)
            measurement model parameter
        :return: 2-D ndarray, of shape (self.xD, in_dim), where in = self.xD (add) or self.xD+self.rD (non-add)
            Jacobian matrix of the measurement model, where the second dimension dependes on the noise additivity.
        """
        raise NotImplementedError

    def dyn_eval(self, xq, pars, dx=False):
        """
        Evaluates system dynamics function according to noise additivity.

        :param xq: 1-D array_like
            augmented system state
        :param pars:
            system dynamics parameters
        :param dx: boolean

        :return:
            if dx == True returns evaluation of the system dynamics Jacobian
            if dx == False returns evaluation of the system dynamics
        """
        if self.q_additive:
            assert len(xq) == self.xD
            if dx:
                out = (self.dyn_fcn_dx(xq, 0, pars).T.flatten())
            else:
                out = self.dyn_fcn(xq, 0, pars)
        else:
            assert len(xq) == self.xD + self.qD
            x, q = xq[:self.xD], xq[-self.qD:]
            if dx:
                out = (self.dyn_fcn_dx(x, q, pars).T.flatten())
            else:
                out = self.dyn_fcn(x, q, pars)
        return out

    def meas_eval(self, xr, pars, dx=False):
        """
        Evaluates measurement model function according to noise additivity.

        :param xr: 1-D array_like
            augmented system state
        :param pars:
            measurement model parameters
        :param dx: boolean

        :return:
            if dx == True returns evaluation of the measurement model Jacobian
            if dx == False returns evaluation of the measurement model
        """
        if self.r_additive:
            assert len(xr) == self.xD
            if dx:
                out = (self.meas_fcn_dx(xr, 0, pars).T.flatten())
            else:
                out = self.meas_fcn(xr, 0, pars)
        else:
            assert len(xr) == self.xD + self.rD
            x, r = xr[:self.xD], xr[-self.rD:]
            if dx:
                out = (self.meas_fcn_dx(x, r, pars).T.flatten())
            else:
                out = self.meas_fcn(x, r, pars)
        return out

    def check_jacobians(self, h=1e-8):
        """
        Checks that both Jacobians are correctly implemented using numerical approximations.
        Prints the errors, user decides whether they're acceptable.

        :param h: step size in derivative approximations
        :return: None
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
        """
        General implementation of the SSM simulation starting from initial conditions for a given number of time steps

        :param steps:
            number of time steps in state trajectory
        :param mc_sims:
            number of trajectories to simulate (the initial state is drawn randomly)
        :return:
            arrays with simulated state trajectories and measurements
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
