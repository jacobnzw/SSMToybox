from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na


class System(object, metaclass=ABCMeta):
    """
    General continuous-time dynamical system

    Attributes
    ----------
    xD : int
        State dimension.

    zD : int
        Measurement dimension.

    qD : int
        State noise dimension.

    rD : int
        Measurement noise dimension.

    q_additive : bool
        Indicates additivity of state noise.

    r_additive : bool
        Indicates additivity of measurement noise.
    """

    xD = None  # state dimension
    zD = None  # measurement dimension
    qD = None  # state noise dimension
    rD = None  # measurement noise dimension

    q_additive = None  # True = state noise is additive, False = non-additive
    r_additive = None

    def __init__(self, **kwargs):
        self.pars = kwargs
        self.zero_q = np.zeros(self.qD)
        self.zero_r = np.zeros(self.rD)

    @abstractmethod
    def dyn_fcn(self, x, q, pars):
        """ System dynamics.

        Abstract method for the system dynamics.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            System state.

        q : (dim_q, ) ndarray
            System noise.

        pars : (dim_par, ) ndarray
            Parameters of the system dynamics.

        Returns
        -------
        : (dim_x, ) ndarray
            System state in the next time step.
        """
        pass

    @abstractmethod
    def meas_fcn(self, x, r, pars):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            System state.

        r : (dim_r, ) ndarray
            Measurement noise.

        pars : (dim_par, ) ndarray
            Parameters of the measurement model.

        Returns
        -------
        : (dim_z, ) ndarray
            Measurement of the state.
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
            Discrete time step.

        Returns
        -------
        : (dim_par, ) ndarray
            Vector of parameters at a given time.
        """
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, q, pars):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            System state.

        q : (dim_q, ) ndarray
            System noise.

        pars : (dim_par, ) ndarray
            Parameters of the system dynamics.

        Returns
        -------
        : (dim_x, dim_x) ndarray
            Jacobian matrix of the system dynamics, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (dim_x, dim_x)
                * non-additive: (dim_x, dim_x + dim_q)
        """
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, pars):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            System state.

        r : (dim_r, ) ndarray
            Measurement noise.

        pars : (dim_par, ) ndarray
            Parameters of the measurement model.

        Returns
        -------
        2-D numpy.ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (dim_x, dim_x)
                * non-additive: (dim_x, dim_x + dim_r)
        """
        pass

    @abstractmethod
    def state_noise_sample(self, size=None):
        """
        Sample from a state noise distribution.

        Parameters
        ----------
        size : int or tuple
            Sample size.

        Returns
        -------
        : (size) ndarray
            Noise samples.
        """
        pass

    @abstractmethod
    def measurement_noise_sample(self, size=None):
        """
        Sample from a measurement noise distribution.

        Parameters
        ----------
        size : int or tuple
            Sample size.

        Returns
        -------
        : (size) ndarray
            Noise samples.
        """
        pass

    @abstractmethod
    def initial_condition_sample(self, size=None):
        """
        Sample from a distribution over the system initial conditions.

        Parameters
        ----------
        size : int or tuple
            Sample size.

        Returns
        -------
        : (size) ndarray
            Initial state samples.
        """
        pass

    def dyn_eval(self, xq, pars, dx=False):
        """Evaluation of the system dynamics according to noise additivity.

        Parameters
        ----------
        xq : (dim_x + dim_q, ) ndarray
            Augmented system state.

        pars : (dim_par, ) ndarray
            System dynamics parameters.

        dx : bool
            Evaluate derivatives
            * ``True``: Evaluates derivatives (Jacobian) of the system dynamics.
            * ``False``: Evaluates system dynamics.

        Returns
        -------
        : (dim_x, ) ndarray
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
        xr : (dim_x + dim_r, ) ndarray
            Augmented system state.

        pars : (dim_par, ) ndarray
            Measurement model parameters.

        dx : bool
            Evaluate derivatives
            * ``True``: Evaluates derivatives (Jacobian) of the measurement model.
            * ``False``: Evaluates measurement model.

        Returns
        -------
        : (dim_z, ) ndarray
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

        Checks that both Jacobians are correctly implemented using numerical approximations. Prints the errors and
        user decides whether they're acceptable.

        Parameters
        ----------
        h : float
            Step size in derivative approximations.
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
        """
        Computes continuous-time system state trajectory using the given ODE integration method.

        Parameters
        ----------
        method : str {'euler', 'rk4'}, optional
            ODE integration method.

        dt : float, optional
            Discretization step.

        duration : int, optional
            Length of trajectory in seconds.

        mc_sims : int, optional
            Number of Monte Carlo simulations.

        Returns
        -------
        : (dim_x, num_time_steps, num_mc_sims) ndarray
            State trajectories of the continuous-time dynamic system.
        """
        # ensure sensible values of dt
        assert dt < duration

        # get ODE integration method
        ode_method = self._get_ode_method(method)

        # allocate space for system state and noise
        steps = int(np.floor(duration / dt))
        x = np.zeros((self.xD, steps+1, mc_sims))
        q = self.state_noise_sample((mc_sims, steps+1))
        x0 = self.initial_condition_sample(mc_sims)  # (D, mc_sims)
        x[:, 0, :] = x0  # store initial states at k=0

        # continuous-time system simulation
        for imc in range(mc_sims):
            for k in range(1, steps+1):
                theta = self.par_fcn(k - 1)
                # computes next state x(t + dt) by ODE integration
                x[:, k, imc] = ode_method(self.dyn_fcn, x[:, k - 1, imc], q[:, k - 1, imc], theta, dt)
        return x[:, 1:, :]

    def simulate_measurements(self, x, mc_per_step=1):
        """
        Simulates measurements

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State trajectory.

        mc_per_step : int, optional
            Number of measurements to generate in each time step.

        Returns
        -------
        : (dim_y, num_time_steps, num_mc_sims) ndarray
            Measurement trajectories of the continuous-time dynamic system.
        """
        # x - state trajectory, freq - sampling frequency [Hz],
        # mc_per_step - how many measurement to generate in each time step

        d, steps = x.shape

        # Generate measurement noise
        r = self.measurement_noise_sample((mc_per_step, steps))
        y = np.zeros((self.zD, steps, mc_per_step))
        for imc in range(mc_per_step):
            for k in range(steps):
                theta = self.par_fcn(k - 1)
                y[:, k, imc] = self.meas_fcn(x[:, k], r[:, k, imc], theta)
        return y

    def _ode_euler(self, func, x, q, theta, dt):
        """
        ODE integration using Euler approximation.

        Parameters
        ----------
        func : function
            Function defining the system dynamics.

        x : (dim_x, ) ndarray
            Previous system state.

        q : (dim_q, ) ndarray
            System (process) noise.

        theta : (dim_par, ) ndarray
            Dynamics parameters.

        dt : float
            Discretization step.

        Returns
        -------
        : (dim_x, ) ndarray
            State in the next time step.
        """
        xdot = func(x, q, theta)
        return x + dt * xdot

    def _ode_rk4(self, func, x, q, theta, dt):
        """
        ODE integration using 4th-order Runge-Kutta approximation.

        Parameters
        ----------
        func : function
            Function defining the system dynamics.

        x : (dim_x, ) ndarray
            Previous system state.

        q : (dim_q, ) ndarray
            System (process) noise.

        theta : (dim_par, ) ndarray
            Dynamics parameters.

        dt : float
            Discretization step.

        Returns
        -------
        : (dim_x, ) ndarray
            State in the next time step.
        """
        dt2 = 0.5 * dt
        k1 = func(x, q, theta)
        k2 = func(x + dt2 * k1, q, theta)
        k3 = func(x + dt2 * k2, q, theta)
        k4 = func(x + dt * k3, q, theta)
        return x + (dt / 6) * (k1 + 2 * (k2 + k3) + k4)

    def _get_ode_method(self, method):
        """
        Get an ODE integration method.

        Parameters
        ----------
        method : str {'euler', 'rk4'}
            ODE integration method.

        Returns
        -------
        : function
            Function handle to the desired ODE integration method.
        """
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


class GaussianSystem(System):
    """
    System where the state and measurement noise are Gaussian.

    Parameters
    ----------
    x0_mean : (dim_x, ) ndarray
        Mean of the initial system state.

    x0_cov : (dim_x, dim_x) ndarray
        Covariance of the initial system state.

    q_mean : (dim_q, ) ndarray
        Mean of the state noise.

    q_cov : (dim_q, dim_q) ndarray
        Covariance of the state noise.

    r_mean : (dim_r, ) ndarray
        Mean of the measurement noise.

    r_cov : (dim_r, dim_r) ndarray
        Covariance of the measurement noise.

    q_gain : (dim_x, dim_q) ndarray
        Gain of the state noise.
    """

    def __init__(self, x0_mean=None, x0_cov=None, q_mean=None, q_cov=None, r_mean=None, r_cov=None, q_gain=None):

        # use default value of statistics for Gaussian SSM if None provided
        kwargs = {
            'x0_mean': x0_mean if not None else np.zeros(self.xD),
            'x0_cov': x0_cov if not None else np.eye(self.xD),
            'q_mean': q_mean if not None else np.zeros(self.qD),
            'q_cov': q_cov if not None else np.eye(self.qD),
            'r_mean': r_mean if not None else np.zeros(self.rD),
            'r_cov': r_cov if not None else np.eye(self.rD),
            'q_gain': q_gain if not None else np.eye(self.qD)
        }
        super(GaussianSystem, self).__init__(**kwargs)

    @abstractmethod
    def dyn_fcn(self, x, q, pars):
        pass

    @abstractmethod
    def meas_fcn(self, x, r, pars):
        pass

    @abstractmethod
    def par_fcn(self, time):
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, q, pars):
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, pars):
        pass

    def state_noise_sample(self, size=None):
        """
        Generate samples of Gaussian state noise.

        Parameters
        ----------
        size : int or tuple
            Sample size.

        Returns
        -------
        : (size) ndarray
            Samples of Gaussian state noise.
        """
        q_mean, q_cov = self.get_pars('q_mean', 'q_cov')
        return np.random.multivariate_normal(q_mean, q_cov, size).T

    def measurement_noise_sample(self, size=None):
        """
        Generate samples of Gaussian measurement noise.

        Parameters
        ----------
        size : int or tuple
            Sample size.

        Returns
        -------
        : (size) ndarray
            Samples of Gaussian measurement noise.
        """
        r_mean, r_cov = self.get_pars('r_mean', 'r_cov')
        return np.random.multivariate_normal(r_mean, r_cov, size).T

    def initial_condition_sample(self, size=None):
        """
        Generate samples of Gaussian initial system state.

        Parameters
        ----------
        size : int or tuple
            Sample size.

        Returns
        -------
        : (size) ndarray
            Samples of Gaussian initial system state.
        """
        x0_mean, x0_cov = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(x0_mean, x0_cov, size).T


class ReentryVehicleRadarTrackingGaussSystem(GaussianSystem):
    """
    Radar tracking of the reentry vehicle as described in [Julier2004]_.
    Vehicle is entering Earth's atmosphere at high altitude and with great speed, ground radar is tracking it.

    State: :math:`\\mathbf{x} = [x, y, \\dot{x}, \\dot{y}, \\omega]`, where
        :math:`x`, :math:`y`
            Position in 2D.
        :math:`\\dot{x}`, :math:`\\dot{y}`
            Velocity in 2D.
        :math:`\\omega`
            Aerodynamic parameter.

    Measurements: :math:`\\mathbf{y} = [r, \\theta]`, where
        :math:`r`
            Range to target.
        :math:`\\theta`
            Bearing to the target.

    References
    ----------
    .. [Julier2004] Julier, S. J., & Uhlmann, J. K. (2004). Unscented Filtering and Nonlinear Estimation.
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
            'q_gain': np.vstack((np.zeros((2, 3)), np.eye(3)))
        }
        super(ReentryVehicleRadarTrackingGaussSystem, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        # scaled ballistic coefficient
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


class ReentryVehicleRadarTrackingSimpleGaussSystem(GaussianSystem):
    """
    Radar tracking of the reentry vehicle as described in [Julier2000]_.
    High velocity projectile is entering atmosphere, radar positioned 100,000ft above Earth's surface (and 100,
    000ft horizontally) is producing range measurements.

    State: :math:`\\mathbf{x} = [y, \\dot{y}, \\omega]`, where
        :math:`y`
            Position.
        :math:`\\dot{y}`
            Velocity.
        :math:`\\omega`
            Aerodynamic parameter.

    Measurements: :math:`y = r`, where
        :math:`r`
            Range to target.

    References
    ----------
    .. [Julier2000] S. J. Julier, J. K. Uhlmann, and H. F. Durrant-Whyte, "A New Method for the Nonlinear Transformation
                    of Means and Covariances in Filters and Estimators," IEEE Transactions on Automatic Control.,
                    vol. 45, no. 3, pp. 477–482, 2000.
    """

    xD = 3
    zD = 1  # measurement dimension
    qD = 3
    rD = 1  # measurement noise dimension
    q_additive = True
    r_additive = True

    R0 = 6371  # Earth's radius [km]  #2.0925e7  # Earth's radius [ft]
    # radar location: 30km (~100k ft) above the surface, radar-to-body horizontal range
    sx, sy = 30, 30
    Gamma = 1/6.096

    def __init__(self):
        kwargs = {
            'x0_mean': np.array([90, 6, 1.5]),  # km, km/s
            'x0_cov': np.diag([0.3048**2, 1.2192**2, 1e-4]),  # km^2, km^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[0.03048**2]]),
            'q_factor': np.vstack(np.eye(3))
        }
        super(ReentryVehicleRadarTrackingSimpleGaussSystem, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([-x[1] + q[0],
                         -np.exp(-self.Gamma * x[0]) * x[1]**2 * x[2] + q[1],
                         q[2]])

    def meas_fcn(self, x, r, pars):
        # range
        rng = np.sqrt(self.sx ** 2 + (x[0] - self.sy) ** 2)
        return np.array([rng]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


def radar_tracking_demo():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    sys = ReentryVehicleRadarTrackingGaussSystem()
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