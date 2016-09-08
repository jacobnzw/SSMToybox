from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na


# Rough, preliminary code up of the continuous-time system simulations
class System(object, metaclass=ABCMeta):
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
        # TODO: if q_factor not given, use identity matrix
        # TODO: if _mean not given, assume zero-mean

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

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
            if dx:
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
            # assert len(xr) == self.xD
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
        print("Errors in Jacobians\n{}\n{}".format(np.abs(jac_fx - self.dyn_eval(xq, par, dx=True)),
                                                   np.abs(jac_hx - self.meas_eval(xr, par, dx=True))))

    def simulate_trajectory(self, method='euler', dt=0.1, duration=10, mc_sims=1):
        # ensure sensible values of dt
        assert dt < duration

        # get the system statistics
        stats = 'x0_mean', 'x0_cov', 'q_mean', 'q_cov'
        x0_mean, x0_cov, q_mean, q_cov = self.get_pars(*stats)

        # get ODE integration method
        ode_method = self._get_ode_method(method)

        # allocate space for system state and noise
        steps = int(np.floor(duration / dt))
        x = np.zeros((self.xD, steps, mc_sims))
        q = np.random.multivariate_normal(q_mean, q_cov, size=(mc_sims, steps)).T
        x0 = np.random.multivariate_normal(x0_mean, x0_cov, size=mc_sims).T  # (D, mc_sims)
        x[:, 0, :] = x0  # store initial states at k=0

        # continuous-time system simulation
        for imc in range(mc_sims):
            for k in range(1, steps):
                theta = self.par_fcn(k - 1)
                # computes next state x(t + dt) by ODE integration
                x[:, k, imc] = ode_method(self.dyn_fcn, x[:, k - 1, imc], q[:, k - 1, imc], theta, dt)
        return x

    def simulate_measurements(self, x, mc_per_step=1):
        # x - state trajectory, freq - sampling frequency [Hz],
        # mc_per_step - how many measurement to generate in each time step

        # get the system statistics
        stats = 'r_mean', 'r_cov'
        r_mean, r_cov = self.get_pars(*stats)
        d, steps = x.shape

        # Generate measurement noise
        r = np.random.multivariate_normal(r_mean, r_cov, (mc_per_step, steps)).T
        y = np.zeros((self.zD, steps, mc_per_step))
        for imc in range(mc_per_step):
            for k in range(1, steps):
                theta = self.par_fcn(k - 1)
                y[:, k, imc] = self.meas_fcn(x[:, k], r[:, k, imc], theta)
        return y


    def _ode_euler(self, func, x, q, theta, dt):
        # Euler ODE integration
        # x-state, q-noise, dt-time increment, func-function handle
        xdot = func(x, q, theta)
        return x + dt * xdot

    def _ode_rk4(self, func, x, q, theta, dt):
        # 4-th order Runge-Kutta ODE integration
        dt2 = 0.5 * dt
        k1 = func(x, q, theta)
        k2 = func(x + dt2 * k1, q, theta)
        k3 = func(x + dt2 * k2, q, theta)
        k4 = func(x + dt * k3, q, theta)
        return x + (dt / 6) * (k1 + 2 * (k2 + k3) + k4)

    def _get_ode_method(self, method):
        method = method.lower()
        if method == 'euler':
            return self._ode_euler
        elif method == 'rk4':
            return self._ode_rk4
        else:
            raise ValueError("Unknown ODE integration method {}".format(method))

    def set_pars(self, key, value):
        self.pars[key] = value

    def get_pars(self, *keys):
        values = []
        for k in keys:
            values.append(self.pars.get(k))
        return values


class ReentryRadar(System):
    """
    Radar tracking of the reentry vehicle as described in [1]_.
    Vehicle is entering Earth's atmosphere at high altitude and with great speed, ground radar is tracking it.

    State
    -----
    [px, py, vx, vy, x5]
    (px, py) - position,
    (vx, vy) - velocity,
    x5 - aerodynamic parameter

    Measurements
    ------------
    range and bearing


    References
    ----------
    .. [1] Julier, S. J., & Uhlmann, J. K. (2004). Unscented Filtering and Nonlinear Estimation.
           Proceedings of the IEEE, 92(3), 401-422

    """

    xD = 5
    zD = 2  # measurement dimension
    qD = 3
    rD = 2  # measurement noise dimension
    q_additive = True
    r_additive = True

    R0 = 6374  # Earth's radius
    H0 = 13.406
    Gm0 = 3.9860e5
    b0 = -0.59783  # balistic coefficient of a typical vehicle
    sx, sy = R0, 0  # radar location

    def __init__(self):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
        kwargs = {
            'x0_mean': np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([1e-6, 1e-6, 1e-6, 1e-6, 0]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array([[2.4064e-4, 0, 0],
                               [0, 2.4064e-4, 0],
                               [0, 0, 0]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[1e-6, 0],
                               [0, 0.17e-3 ** 2]]),
            'q_factor': np.vstack((np.zeros((2, 3)), np.eye(3)))
        }
        super(ReentryRadar, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        # scaled balistic coefficient
        b = self.b0 * np.exp(x[4])
        # distance from center of the Earth
        R = np.sqrt(x[0] ** 2 + x[1] ** 2)
        # speed
        V = np.sqrt(x[2] ** 2 + x[3] ** 2)
        # drag force
        D = b * np.exp((self.R0 - R) / self.H0) * V
        # gravity force
        G = -self.Gm0 / R ** 3
        return np.array([x[2],
                         x[3],
                         D * x[2] + G * x[0] + q[0],
                         D * x[3] + G * x[1] + q[1],
                         q[2]])

    def meas_fcn(self, x, r, pars):
        # range
        rng = np.sqrt((x[0] - self.sx) ** 2 + (x[1] - self.sy) ** 2)
        # bearing
        theta = np.arctan2((x[1] - self.sy), (x[0] - self.sx))
        return np.array([rng, theta]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


def radar_tracking_demo():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    sys = ReentryRadar()
    mc = 10
    x = sys.simulate_trajectory(method='rk4', dt=0.05, duration=200, mc_sims=mc)
    y = sys.simulate_measurements(x[..., 0])

    plt.figure()
    g = GridSpec(2, 4)
    plt.subplot(g[:, :2])
    # Earth surface w/ radar position
    t = 0.02 * np.arange(-1, 4, 0.1)
    plt.plot(sys.R0 * np.cos(t), sys.R0 * np.sin(t), 'darkblue', lw=2)
    plt.plot(sys.sx, sys.sy, 'ko')
    # vehicle trajectory
    for i in range(mc):
        plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='r', ls='--')
    plt.subplot(g[:, 2:], polar=True)
    plt.plot((y[1, :, 0]), y[0, :, 0], 'ko')
    plt.show()


if __name__ == '__main__':
    radar_tracking_demo()