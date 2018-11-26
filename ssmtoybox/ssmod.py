from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import newaxis as na
from ssmtoybox.utils import multivariate_t


"""
Transition models
"""


class TransitionModel(metaclass=ABCMeta):
    """State transition model

    Definition of both discrete and continuous-time dynamics with simulation.

    Parameters
    ----------
    init_rv : RandomVariable
        Distribution of the system initial conditions.

    noise_rv : RandomVariable
        Distribution of the process (state) noise.

    noise_gain : (dim_out, dim_noise) ndarray, optional
        Noise gain matrix.

    Attributes
    ----------
    dim_in : int
        Input dimension of the state transition function.

    dim_out : int
        Output dimension of the state transition function.

    dim_noise : int
        Dimensionality of the process noise vector.

    noise_additive : bool
        Indicates additivity of the noise. `True` if noise is additive, `False` otherwise.
    """

    dim_in = None
    dim_out = None
    dim_noise = None
    noise_additive = None

    def __init__(self, init_rv, noise_rv, noise_gain=None):
        # distribution of initial conditions
        self.init_rv = init_rv
        # distribution of process noise
        self.noise_rv = noise_rv
        # zero vec for convenience
        self.zero_q = np.zeros(self.dim_noise)  # TODO rename to q_zero
        if noise_gain is None:
            self.noise_gain = np.eye(self.dim_out, self.dim_noise)

    @abstractmethod
    def dyn_fcn(self, x, q, time):
        """Discrete-time system dynamics.

        Abstract method for the discrete-time system dynamics.

        Parameters
        ----------
        x : (dim_in, ) ndarray
            System state.

        q : (dim_noise, ) ndarray
            System noise.

        time : int
            Time index.

        Returns
        -------
        : (dim_out, ) ndarray
             System state in the next time step.
        """
        pass

    @abstractmethod
    def dyn_fcn_cont(self, x, q, time):
        """Continuous-time system dynamics.

        Abstract method for the continuous-time system dynamics.

        Parameters
        ----------
        x : (dim_in, ) ndarray
            System state.

        q : (dim_noise, ) ndarray
            System noise.

        time : int
            Time index.

        Returns
        -------
        : (dim_out, ) ndarray
           Time-derivative of system state evaluated at given state and time
        """
        pass

    @abstractmethod
    def dyn_fcn_dx(self, x, r, time):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : (dim_in, ) ndarray
            System state.

        q : (dim_noise, ) ndarray
            System noise.

        time : int
            Time index.

        Returns
        -------
        : (dim_out, dim_in) ndarray
            Jacobian matrix of the system dynamics. Note that in non-additive noise case `dim_in = dim_out + dim_noise`.
        """
        pass

    def dyn_eval(self, xq, time, dx=False):
        """Evaluation of the system dynamics according to noise additivity.

        Parameters
        ----------
        xq : (dim_in + dim_noise) ndarray
            Augmented system state.

        time : int
            Time index.

        dx : bool, optional
            * ``True``: Evaluates derivative (Jacobian) of the system dynamics
            * ``False``: Evaluates system dynamics

        Returns
        -------
        : (dim_out, )
            Evaluated system dynamics, if `dx == False`.

        : (dim_out, dim_in)
            Evaluated Jacobian of the system dynamics, if `dx == True`.
        """

        if self.noise_additive:
            assert len(xq) == self.dim_in
            if dx:
                out = (self.dyn_fcn_dx(xq, self.zero_q, time).T.flatten())
            else:
                out = self.dyn_fcn(xq, self.zero_q, time)
        else:
            assert len(xq) == self.dim_in + self.dim_noise
            x, q = xq[:self.dim_in], xq[-self.dim_noise:]
            if dx:
                out = (self.dyn_fcn_dx(x, q, time).T.flatten())
            else:
                out = self.dyn_fcn(x, q, time)
        return out

    def simulate_discrete(self, steps, mc_sims=1):
        """Simulation of discrete-time state trajectory.

        Simulation starts from initial conditions for a given number of time steps

        Parameters
        ----------
        steps : int
            Number of time steps in state trajectory

        mc_sims : int
            Number of trajectories to simulate (the initial state is drawn randomly)

        Returns
        -------
        (dim_out, steps, mc_sims) ndarray
            The `mc_sims` state trajectories, each `steps` long.
        """

        # allocate space for state and measurement sequences
        x = np.zeros((self.dim_in, steps, mc_sims))
        # generate initial conditions, store initial states at k=0
        x[:, 0, :] = self.init_rv.sample(mc_sims)  # (D, mc_sims)

        # generate state and measurement noise
        q = self.noise_rv.sample((steps, mc_sims))

        # simulate SSM `mc_sims` times for `steps` time steps
        for imc in range(mc_sims):
            for k in range(1, steps):
                x[:, k, imc] = self.dyn_fcn(x[:, k-1, imc], q[:, k-1, imc], k-1)
        return x

    def simulate_continuous(self, duration, dt=0.1, mc_sims=1, method='euler'):
        """
        Computes continuous-time system state trajectory using the given ODE integration method.

        Parameters
        ----------
        duration : int
            Length of trajectory in seconds.

        dt : float, optional
            Discretization step.

        mc_sims : int, optional
            Number of Monte Carlo simulations.

        method : str {'euler', 'rk4'}, optional
            ODE integration method.

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
        x = np.zeros((self.dim_in, steps+1, mc_sims))
        # sample initial conditions and process noise
        x[:, 0, :] = self.init_rv.sample(mc_sims)  # (D, mc_sims)
        q = self.noise_rv.sample((mc_sims, steps + 1))

        # continuous-time system simulation
        for imc in range(mc_sims):
            for k in range(1, steps+1):
                # computes next state x(t + dt) by ODE integration
                x[:, k, imc] = ode_method(self.dyn_fcn, x[:, k - 1, imc], q[:, k - 1, imc], k-1, dt)
        return x[:, 1:, :]

    @staticmethod
    def ode_euler(func, x, q, time, dt):
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

        time : (dim_par, ) ndarray
            Time index.

        dt : float
            Discretization step.

        Returns
        -------
        : (dim_x, ) ndarray
            State in the next time step.
        """
        xdot = func(x, q, time)
        return x + dt * xdot

    @staticmethod
    def ode_rk4(func, x, q, time, dt):
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

        time : (dim_par, ) ndarray
            Time index.

        dt : float
            Discretization step.

        Returns
        -------
        : (dim_x, ) ndarray
            State in the next time step.
        """
        dt2 = 0.5 * dt
        k1 = func(x, q, time)
        k2 = func(x + dt2 * k1, q, time)
        k3 = func(x + dt2 * k2, q, time)
        k4 = func(x + dt * k3, q, time)
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
            return self.ode_euler
        elif method == 'rk4':
            return self.ode_rk4
        else:
            raise ValueError("Unknown ODE integration method {}".format(method))


class UNGMTransition(TransitionModel):
    """
    Univariate Nonlinear Growth Model (UNGM) with additive noise.

    Notes
    -----
    The model is

    .. math::
        x_{k+1} = 0.5 x_k * \frac{25 x_k}{1 + x_k^2} + 8*\cos(1.2 k) + q_k

    Typically used with :math:`x_0 ~ N(0, 1)`, :math:`q_k ~ N(0, 10)`.
    """

    dim_in = 1
    dim_out = 1
    dim_noise = 1
    noise_additive = True

    def __init__(self, init_rv, noise_rv):
        super(UNGMTransition, self).__init__(init_rv, noise_rv)

    def dyn_fcn(self, x, q, time):
        return np.asarray(0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * time)) + q

    def dyn_fcn_dx(self, x, q, time):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2])

    def dyn_fcn_cont(self, x, q, time):
        pass


class UNGMNATransition(TransitionModel):
    """
    Univariate Nonlinear Growth Model (UNGM) with non-additive noise.

    Notes
    -----
    The model is

    .. math::
        x_{k+1} = 0.5 x_k \frac{25 x_k}{1 + x_k^2} + 8 q_k \cos(1.2 k)

    Typically used with :math:`x_0 ~ N(0, 1)`, :math:`q_k ~ N(0, 10)`.
    """

    dim_in = 1
    dim_out = 1
    dim_noise = 1
    noise_additive = False

    def __init__(self, init_rv, noise_rv):
        super(UNGMNATransition, self).__init__(init_rv, noise_rv)

    def dyn_fcn(self, x, q, time):
        return np.asarray(0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * q[0] * np.cos(1.2 * time))

    def dyn_fcn_cont(self, x, q, time):
        pass

    def dyn_fcn_dx(self, x, q, time):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * time)])


class Pendulum2DTransition(TransitionModel):
    """
    Pendulum with unit length and mass in 2D [1]_ (Example 5.1).

    State
    -----
    `x[0]`: :math:`\alpha` angle from the perpendicular and the direction of the pendulum
    `x[1]`: :math:`\dot{\alpha}` angular speed

    Notes
    -----

    .. math ::

        \begin{bmatrix}
            \alpha_{k+1} \\
            \dot{\alpha}_{k+1}
        \end{bmatrix} =
        \begin{bmatrix}
            \alpha_k + \dot{\alpha}_k\Delta t \\
            \dot{\alpha}_k - g\sin(\alpha_k)\Delta t
        \end{bmatrix} + \mathbf{q}_k

    References
    ----------
    .. [1] Sarkka, S., Bayesian Filtering and Smoothing, Cambridge University Press, 2013.
    """

    dim_in = 2
    dim_out = 2
    dim_noise = 2
    noise_additive = True

    g = 9.81  # gravitational acceleration

    def __init__(self, init_rv, noise_rv, dt=0.01):
        super(Pendulum2DTransition, self).__init__(init_rv, noise_rv)
        self.dt = dt

    def dyn_fcn(self, x, q, time):
        return np.array([x[0] + x[1] * self.dt, x[1] - self.g * self.dt * np.sin(x[0])]) + q

    def dyn_fcn_cont(self, x, q, time):
        pass

    def dyn_fcn_dx(self, x, r, time):
        return np.array([[1.0, self.dt],
                         [-self.g * self.dt * np.cos(x[0]), 1.0]])


class ReentryVehicle1DTransition(TransitionModel):
    """
    Simplified model of reentry vehicle.

    "The position, velocity and constant ballistic coefficient of a body as it re-enters the atmosphere at a very high
    altitude at a very high velocity. Acceleration due to gravity is negligible compared to the altitude and
    velocity-dependent drag terms. The body is constrained so that it falls vertically." [1]_

    State
    -----
    :math:`\\mathbf{x} = [y, \\dot{y}, \\omega]`, where
        :math:`y`
            Altitude.

        :math:`\\dot{y}`
            Velocity.

        :math:`\\omega`
            (constant) ballistic coefficient.

    Notes
    -----
    # TODO: process equation, reasonable statistics


    References
    ----------
    .. [1] Julier, S. and Uhlmann, J., A General Method for Approximating Nonlinear Transformations of Probability
           Distributions, 1996
    """

    dim_in = 3
    dim_out = 3
    dim_noise = 3
    noise_additive = True

    def __init__(self, init_rv, noise_rv, dt=0.1):
        super(ReentryVehicle1DTransition, self).__init__(init_rv, noise_rv)
        self.dt = dt
        self.Gamma = 1 / 6.096

    def dyn_fcn(self, x, q, time):
        return np.array([x[0] - self.dt * x[1] + q[0],
                         x[1] - self.dt * np.exp(-self.Gamma * x[0]) * x[1] ** 2 * x[2] + q[1],
                         x[2] + q[2]])

    def dyn_fcn_cont(self, x, q, time):
        return np.array([-x[1] + q[0],
                         -np.exp(-self.Gamma * x[0]) * x[1] ** 2 * x[2] + q[1],
                         q[2]])

    def dyn_fcn_dx(self, x, r, time):
        pass


class ReentryVehicle2DTransition(TransitionModel):
    """
    Reentry vehicle entering the atmosphere at high altitude and at a very speed.

    "This type of problem has been identified by a number of authors [2]_-[5]_ as being particularly stressful for
    filters and trackers because of the strong nonlinearities exhibited by the forces which act on the vehicle. There
    are three types of forces in effect. The most dominant is aerodynamic drag, which is a function of vehicle speed
    and has a substantial nonlinear variation in altitude. The second type of force is gravity, which accelerates the
    vehicle toward the center of the earth. The final forces are random buffeting terms." [1]_

    "The tracking problem is made more difficult by the fact that the drag properties of the vehicle might be only very
    crudely known." [1]_

    The model is specified in Cartesian `geocentric coordinates <https://en.wikipedia.org/wiki/ECEF>`.

    State
    -----
    :math:`\\mathbf{x} = [x, y, \\dot{x}, \\dot{y}, \\omega]`, where
        :math:`x`, :math:`y`
            Position in 2D.

        :math:`\\dot{x}`, :math:`\\dot{y}`
            Velocity in 2D.

        :math:`\\omega`
            Aerodynamic parameter.

    Notes
    -----
    # TODO: process equation, reasonable stats

    x_0 ~ N(0, P_0), q ~ N(0, Q)
    kwargs = {
            'x0_mean': np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': self.dt ** -1 * np.array(
                [[2.4064e-5, 0, 0],
                 [0, 2.4064e-5, 0],
                 [0, 0, 1e-6]]),
            'q_gain': np.vstack((np.zeros((2, 3)), np.eye(3)))
        }

    .. math::

        P_0 = \diag([  ])

    References
    ----------
    .. [1] Julier, S. and Uhlmann, J. Unscented Filtering and Nonlinear Estimation, Proceedings of IEEE, 2004
    .. [2] P. J. Costa, “Adaptive model architecture and extended Kalman–Bucy filters,”
           IEEE Trans. Aerosp. Electron. Syst., vol. 30, pp. 525–533, Apr. 1994.
    .. [3] M. Athans, R. P. Wishner, and A. Bertolini, “Suboptimal state estimation for continuous-time nonlinear
           systems from discrete noisy measurements,” IEEE Trans. Automat. Contr., vol. AC-13, pp. 504–518, Oct. 1968.
    .. [4] J. W. Austin and C. T. Leondes, “Statistically linearized estimation of reentry trajectories,”
           IEEE Trans. Aerosp. Electron. Syst., vol. AES-17, pp. 54–61, Jan. 1981.
    .. [5] R. K. Mehra, “A comparison of several nonlinear filters for reentry vehicle tracking,”
           IEEE Trans. Automat. Contr., vol. AC-16, pp. 307–319, Aug. 1971.
    """

    dim_in = 5
    dim_out = 5
    dim_noise = 3
    noise_additive = True

    def __init__(self, init_rv, noise_rv, dt=0.1):
        super(ReentryVehicle2DTransition, self).__init__(init_rv, noise_rv)
        self.dt = dt
        self.R0 = 6374  # Earth's radius
        self.H0 = 13.406
        self.Gm0 = 3.9860e5
        self.b0 = -0.59783  # ballistic coefficient of a typical vehicle

    def dyn_fcn(self, x, q, time):
        """
        Equation describing dynamics of the reentry vehicle.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        q : (dim_q, ) ndarray
            State noise vector.

        time :
            Time index.

        Returns
        -------
        : (dim_x, ) ndarray
            System state in the next time step.
        """
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
        return np.array([x[0] + self.dt * x[2],
                         x[1] + self.dt * x[3],
                         x[2] + self.dt * (D * x[2] + G * x[0]) + q[0],
                         x[3] + self.dt * (D * x[3] + G * x[1]) + q[1],
                         x[4] + q[2]])

    def dyn_fcn_dx(self, x, r, time):
        pass

    def dyn_fcn_cont(self, x, q, time):
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


"""
Measurement models
"""


class MeasurementModel(metaclass=ABCMeta):
    """Measurement model.

    Describes transformation of the system state into measurement.

    Parameters
    ----------
    noise_rv : RandomVariable
        Distribution of the measurement noise.


    Attributes
    ----------
    dim_in : int
        Input dimension of the state transition function.

    dim_out : int
        Output dimension of the state transition function.

    dim_noise : int
        Dimensionality of the process noise vector.

    noise_additive : bool
        Indicates additivity of the measurement noise. `True` if noise is additive, `False` otherwise.
    """

    # TODO use index mask to pick out states to use for computing the measurement.
    # TODO should represent effective dim. to verify the index mask (if proper # dims were selected).
    dim_in = None
    dim_out = None
    dim_noise = None
    noise_additive = None

    def __init__(self, noise_rv):
        # distribution of process noise
        self.noise_rv = noise_rv
        # zero vec for convenience
        self.zero_r = np.zeros(self.dim_noise)  # TODO: rename to r_zero

    @abstractmethod
    def meas_fcn(self, x, r, time):
        """Measurement model.

        Abstract method for the measurement model.

        Parameters
        ----------
        x : (dim_in, ) ndarray  # TODO: there will be more to say once the mask is implemented
            System state.
        
        r : (dim_noise, ) ndarray
            Measurement noise.
        
        time : int
            Time index.

        Returns
        -------
        : (dim_out, ) ndarray
            Measurement of the state.
        """
        pass

    @abstractmethod
    def meas_fcn_dx(self, x, r, time):
        """Jacobian of the measurement model.

        Abstract method for the Jacobian of measurement model. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : (dim_in, ) ndarray
            System state.
        
        r : (dim_noise, ) ndarray
            Measurement noise.
        
        time : int
            Time index.

        Returns
        -------
        : (dim_out, dim_in) ndarray
            Jacobian matrix of the system dynamics. Note that in non-additive noise case `dim_in = dim_out + dim_noise`.
        """
        pass

    def meas_eval(self, xr, time, dx=False):
        """Evaluation of the measurement model according to noise additivity.

        Parameters
        ----------
        xr : (dim_in + dim_noise) ndarray
            System state augmented with the measurement noise.

        time : int
            Time index.

        dx : bool, optional
            * ``True``: Evaluates derivative (Jacobian) of the measurement model.
            * ``False``: Evaluates measurement model.

        Returns
        -------
        : (dim_out, )
            Evaluated measurement model, if `dx == False`.

        : (dim_out, dim_in)
            Evaluated Jacobian of the measurement model, if `dx == True`.
        """

        if self.noise_additive:
            # assert len(xr) == self.xD
            if dx:
                out = (self.meas_fcn_dx(xr, self.zero_r, time).T.flatten())
            else:
                out = self.meas_fcn(xr, self.zero_r, time)
        else:
            assert len(xr) == self.dim_in + self.dim_noise
            x, r = xr[:self.dim_in], xr[-self.dim_noise:]
            if dx:
                out = (self.meas_fcn_dx(x, r, time).T.flatten())
            else:
                out = self.meas_fcn(x, r, time)
        return out

    def simulate_measurements(self, x):
        """
        Simulates measurements

        Parameters
        ----------
        x : (dim_x, steps, mc_sims) ndarray
            State trajectory.

        Returns
        -------
        : (dim_y, steps, mc_sims) ndarray
            Measurement trajectories of the continuous-time dynamic system.
        """

        d, steps, mc_sims = x.shape

        # Generate measurement noise
        r = self.noise_rv.sample((steps, mc_sims))
        y = np.zeros((self.dim_out, steps, mc_sims))
        for imc in range(mc_sims):
            for k in range(steps):
                y[:, k, imc] = self.meas_fcn(x[:, k, imc], r[:, k, imc], k+1)  # TODO: check time index
        return y


class UNGMMeasurement(MeasurementModel):

    dim_in = 1
    dim_out = 1
    dim_noise = 1
    noise_additive = True

    def __init__(self, noise_rv):
        super(UNGMMeasurement, self).__init__(noise_rv)

    def meas_fcn(self, x, r, time):
        return np.asarray([0.05 * x[0] ** 2]) + r

    def meas_fcn_dx(self, x, r, time):
        return np.asarray([0.1 * x[0]])


class UNGMNAMeasurement(MeasurementModel):

    dim_in = 1
    dim_out = 1
    dim_noise = 1
    noise_additive = False

    def __init__(self, noise_rv):
        super(UNGMNAMeasurement, self).__init__(noise_rv)

    def meas_fcn(self, x, r, time):
        return np.asarray([0.05 * r[0] * x[0] ** 2])

    def meas_fcn_dx(self, x, r, time):
        return np.asarray([0.1 * r[0] * x[0], 0.05 * x[0] ** 2])


class Pendulum2DMeasurement(MeasurementModel):

    dim_in = 2
    dim_out = 1
    dim_noise = 1
    noise_additive = True

    def __init__(self, noise_rv):
        super(Pendulum2DMeasurement, self).__init__(noise_rv)

    def meas_fcn(self, x, r, time):
        return np.array([np.sin(x[0])]) + r

    def meas_fcn_dx(self, x, r, time):
        return np.array([np.cos(x[0]), 0.0])


class RangeMeasurement(MeasurementModel):

    dim_in = 3
    dim_out = 1
    dim_noise = 1
    noise_additive = True

    def __init__(self, noise_rv):
        super(RangeMeasurement, self).__init__(noise_rv)
        self.sx = 30
        self.sy = 30

    def meas_fcn(self, x, r, time):
        rng = np.sqrt(self.sx ** 2 + (x[0] - self.sy) ** 2)
        return np.array([rng]) + r

    def meas_fcn_dx(self, x, r, time):
        pass


class Radar2DMeasurement(MeasurementModel):
    """
    kwargs = {
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[1e-6, 0],
                               [0, 0.17e-3 ** 2]]),
        }

    # TODO: could be extended to 3D + (optionally) range rate measurements
    """

    dim_in = 2
    dim_out = 2
    dim_noise = 2
    noise_additive = True

    def __init__(self, init_dist, noise_rv, radar_loc=None):
        super(Radar2DMeasurement, self).__init__(init_dist, noise_rv)
        # set default radar location
        if radar_loc is None:
            self.radar_loc = np.array([0, 0])

    def meas_fcn(self, x, r, pars):
        """
        Range and bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        r : (dim_r, ) ndarray
            Measurement noise vector.

        pars : tuple
            Unused.

        Returns
        -------
        : (dim_y, ) ndarray
            Range and bearing measurements.
        """
        # range
        rng = np.sqrt((x[0] - self.radar_loc[0]) ** 2 + (x[1] - self.radar_loc[1]) ** 2)
        # bearing
        theta = np.arctan2((x[1] - self.radar_loc[1]), (x[0] - self.radar_loc[0]))
        return np.array([rng, theta]) + r

    def meas_fcn_dx(self, x, r, time):
        pass


class StateSpaceModel(metaclass=ABCMeta):
    """
    Base class for all state-space models.


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

    @abstractmethod
    def state_noise_sample(self, size=None):
        """
        Sample from a state noise distribution.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

        """
        pass

    @abstractmethod
    def measurement_noise_sample(self, size=None):
        """
        Sample from a measurement noise distribution.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

        """
        pass

    @abstractmethod
    def initial_condition_sample(self, size=None):
        """
        Sample from a distribution over the system initial conditions.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------

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

        # allocate space for Jacobians
        fph = np.zeros((self.xD, nq))
        hph = np.zeros((self.zD, nr))
        fmh, hmh = fph.copy(), hph.copy()

        # approximate Jacobians by central differences
        par = self.par_fcn(1.0)
        for i in range(nq):
            fph[:, i] = self.dyn_eval(xqph[:, i], par)
            fmh[:, i] = self.dyn_eval(xqmh[:, i], par)
        for i in range(nr):
            hph[:, i] = self.meas_eval(xrph[:, i], par)
            hmh[:, i] = self.meas_eval(xrmh[:, i], par)
        jac_fx = (2 * h) ** -1 * (fph - fmh)
        jac_hx = (2 * h) ** -1 * (hph - hmh)

        jac_fx_eval = self.dyn_eval(xq, par, dx=True).reshape(self.xD, self.xD)
        jac_hx_eval = self.meas_eval(xr, par, dx=True).reshape(self.zD, self.xD)

        # report approximation error
        print()
        print("Errors in Jacobians")
        print(np.abs(jac_fx - jac_fx_eval))
        print()
        print(np.abs(jac_hx - jac_hx_eval))
        print()

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

        # allocate space for state and measurement sequences
        x = np.zeros((self.xD, steps, mc_sims))
        z = np.zeros((self.zD, steps, mc_sims))

        # generate state and measurement noise
        q = self.state_noise_sample((mc_sims, steps))
        r = self.measurement_noise_sample((mc_sims, steps))

        # generate initial conditions, store initial states at k=0
        x0 = self.initial_condition_sample(mc_sims)  # (D, mc_sims)
        x[:, 0, :] = x0

        # simulate SSM `mc_sims` times for `steps` time steps
        for imc in range(mc_sims):
            for k in range(1, steps):
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


class GaussianStateSpaceModel(StateSpaceModel):
    """
    State-space model with Gaussian noises and initial conditions.

    Parameters
    ----------
    x0_mean : ndarray
        Mean of the state at initial time step.

    x0_cov : ndarray
        Covariance of the state at initial time step.

    q_mean : ndarray
        Mean of the state (process) noise.

    q_cov : ndarray
        Covariance of the state (process) noise.

    r_mean : ndarray
        Mean of the measurement noise.

    r_cov : ndarray
        Covariance of the measurement noise.

    q_gain : ndarray
        Gain of the state (process) noise.
    """

    def __init__(self, x0_mean=None, x0_cov=None, q_mean=None, q_cov=None, r_mean=None, r_cov=None, q_gain=None):

        # use default value of statistics for Gaussian SSM if None provided
        # TODO: sensible defaults differ on case by case basis => specify defaults in subclasses
        kwargs = {
            'x0_mean': x0_mean if x0_mean is not None else np.zeros(self.xD),
            'x0_cov': x0_cov if x0_cov is not None else np.eye(self.xD),
            'q_mean': q_mean if q_mean is not None else np.zeros(self.qD),
            'q_cov': q_cov if q_cov is not None else np.eye(self.qD),
            'r_mean': r_mean if r_mean is not None else np.zeros(self.rD),
            'r_cov': r_cov if r_cov is not None else np.eye(self.rD),
            'q_gain': q_gain if q_gain is not None else np.eye(self.qD)
        }
        super(GaussianStateSpaceModel, self).__init__(**kwargs)

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
        : (dim_y, ) ndarray
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
        : (dim_x, dim_x) or (dim_x, dim_x + dim_q) ndarray
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
        : (dim_x, dim_x) or (dim_x, dim_x + dim_r) ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (dim_x, dim_x)
                * non-additive: (dim_x, dim_x + dim_r)
        """
        pass

    def state_noise_sample(self, size=None):
        """
        Samples of the multivariate Gaussian state noise.

        Parameters
        ----------
        size : int or tuple, optional
            Shape of the returned array of noise samples.

        Returns
        -------
        : float or (size) ndarray
            Samples of the Gaussian state (process) noise.
        """
        q_mean, q_cov = self.get_pars('q_mean', 'q_cov')
        return np.random.multivariate_normal(q_mean, q_cov, size).T

    def measurement_noise_sample(self, size=None):
        """
        Samples of the multivariate Gaussian measurement noise.

        Parameters
        ----------
        size : int or tuple, optional
            Shape of the returned array of noise samples.

        Returns
        -------
        : float or (size) ndarray
            Samples of the Gaussian state (process) noise.
        """
        r_mean, r_cov = self.get_pars('r_mean', 'r_cov')
        return np.random.multivariate_normal(r_mean, r_cov, size).T

    def initial_condition_sample(self, size=None):
        """
        Samples of the multivariate Gaussian initial state.

        Parameters
        ----------
        size : int or tuple, optional
            Shape of the returned array of noise samples.

        Returns
        -------
        : float or (size) ndarray
            Samples of the initial system state.
        """
        x0_mean, x0_cov = self.get_pars('x0_mean', 'x0_cov')
        return np.random.multivariate_normal(x0_mean, x0_cov, size).T


class StudentStateSpaceModel(StateSpaceModel):

    def __init__(self, x0_mean=None, x0_cov=None, x0_dof=None, q_mean=None, q_cov=None, q_dof=None, q_gain=None,
                 r_mean=None, r_cov=None, r_dof=None):
        """
        State-space model where the state and measurement noises and the initial conditions are Student's t-distributed.

        Parameters
        ----------
        x0_mean : (dim_x, ) ndarray
            Mean of the state at initial time step.

        x0_cov : (dim_x, dim_x) ndarray
            Covariance of the state at initial time step.

        q_mean : (dim_q, ) ndarray
            Mean of the state (process) noise.

        q_cov : (dim_q, dim_q) ndarray
            Covariance of the state (process) noise.

        r_mean : (dim_r, ) ndarray
            Mean of the measurement noise.

        r_cov : (dim_r, dim_r) ndarray
            Covariance of the measurement noise.

        q_gain : (dim_x, dim_q) ndarray
            Gain of the state (process) noise.
        """
        kwargs = {
            'x0_mean': x0_mean if x0_mean is not None else np.zeros(self.xD),
            'x0_cov': x0_cov if x0_cov is not None else np.eye(self.xD),
            'x0_dof': x0_dof if x0_dof is not None else 4.0,  # desired DOF
            'q_mean': q_mean if q_mean is not None else np.zeros(self.qD),
            'q_cov': q_cov if q_cov is not None else np.eye(self.qD),
            'q_gain': q_gain if q_gain is not None else np.eye(self.qD),
            'q_dof': q_dof if q_dof is not None else 4.0,
            'r_mean': r_mean if r_mean is not None else np.zeros(self.rD),
            'r_cov': r_cov if r_cov is not None else np.eye(self.rD),
            'r_dof': r_dof if r_dof is not None else 4.0,
        }
        super(StudentStateSpaceModel, self).__init__(**kwargs)

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
        : (dim_y, ) ndarray
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
        : (dim_x, dim_x) or (dim_x, dim_x + dim_q) ndarray
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
        : (dim_x, dim_x) or (dim_x, dim_x + dim_r) ndarray
            Jacobian matrix of the measurement model, where the second dimension depends on the noise additivity.
            The shape depends on whether or not the state noise is additive. The two cases are:
                * additive: (dim_x, dim_x)
                * non-additive: (dim_x, dim_x + dim_r)
        """
        pass

    def state_noise_sample(self, size=None):
        """
        Samples of the multivariate Student's state noise.

        Parameters
        ----------
        size : int or tuple, optional
            Shape of the returned array of noise samples.

        Returns
        -------
        : float or (size) ndarray
            Samples of the Student's t-distributed state (process) noise.
        """
        q_mean, q_cov, q_dof = self.get_pars('q_mean', 'q_cov', 'q_dof')
        return multivariate_t(q_mean, q_cov, q_dof, size).T

    def measurement_noise_sample(self, size=None):
        """
        Samples of the multivariate Student's measurement noise.

        Parameters
        ----------
        size : int or tuple, optional
            Shape of the returned array of noise samples.

        Returns
        -------
        : float or (size) ndarray
            Samples of the Student's t-distributed measurement noise.
        """
        r_mean, r_cov, r_dof = self.get_pars('r_mean', 'r_cov', 'r_dof')
        return multivariate_t(r_mean, r_cov, r_dof, size).T

    def initial_condition_sample(self, size=None):
        """
        Samples of the multivariate Student's initial state.

        Parameters
        ----------
        size : int or tuple, optional
            Shape of the returned array of noise samples.

        Returns
        -------
        : float or (size) ndarray
            Samples of the Student's t-distributed system initial state.
        """
        x0_mean, x0_cov, x0_dof = self.get_pars('x0_mean', 'x0_cov', 'x0_dof')
        return multivariate_t(x0_mean, x0_cov, x0_dof, size).T


class FrequencyDemodulationGaussSSM(GaussianStateSpaceModel):
    """
    Frequency demodulation SSM from [Pakki]_.

    The objective is to estimate the frequency message :math:`x_1 = \\omega` from noisy in-phase and quadrature
    observations.

    References
    ----------
    .. [Pakki] Pakki, K., et al., Cubature Information Filter and its Applications, Proceedings of the ACC, 2011
    """

    xD = 2
    zD = 2
    qD = 2
    rD = 2

    q_additive = True
    r_additive = True

    # system (process) parameters
    mu = 0.9
    lam = 0.99

    def __init__(self):
        kwargs = {
            'x0_mean': np.zeros(self.xD),  # strange to have zero initial frequency
            'x0_cov': np.eye(self.xD),
            'q_mean': np.zeros(self.qD),
            'q_cov': np.eye(self.qD),
            'r_mean': np.zeros(self.rD),
            'r_cov': 2e-3 * np.eye(self.rD),
            'q_gain': np.eye(self.qD),
        }
        super(FrequencyDemodulationGaussSSM, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([self.mu * x[0], np.arctan(self.lam * x[1] + x[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.array([np.cos(x[1]), np.sin(x[1])]) + r

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def par_fcn(self, time):
        pass


class VanDerPolOscillator2DGaussSSM(GaussianStateSpaceModel):
    """
    Van der Pol oscillator in 2D.

    """

    xD = 2  # state dimension
    zD = 1  # measurement dimension
    qD = 2  # state noise dimension
    rD = 1  # measurement noise dimension

    q_additive = True  # True = state noise is additive, False = non-additive
    r_additive = True

    def __init__(self, dt=0.1, alpha=1):
        self.dt = dt
        self.alpha = alpha
        req_kwargs = {
            'x0_mean': np.zeros(self.xD),
            'x0_cov': np.diag([6.3e-4, 2.2e-4]),
            'q_mean': np.zeros(self.qD),
            'q_cov': np.diag([2.62e-2, 8e-3]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[3e-3]]),
            'q_gain': np.eye(2)
        }
        super(VanDerPolOscillator2DGaussSSM, self).__init__(**req_kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] + self.dt*x[1], x[1] + self.dt*(self.alpha*x[1]*(1 - x[0]**2) - x[0])])

    def meas_fcn(self, x, r, pars):
        return np.array([x[1]]) + r

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass

    def par_fcn(self, time):
        pass


class PendulumGaussSSM(GaussianStateSpaceModel):
    """
    Pendulum in 2D.
    """

    xD = 2  # state dimension
    zD = 1  # measurement dimension
    qD = 2
    rD = 1

    q_additive = True
    r_additive = True

    g = 9.81  # gravitation constant

    def __init__(self, x0_mean=np.array([1.5, 0]), x0_cov=0.01 * np.eye(2), r_cov=np.array([[0.1]]), dt=0.01):
        self.dt = dt
        assert x0_mean.shape == (self.xD,) and x0_cov.shape == (self.xD, self.xD)
        assert r_cov.shape == (self.zD, self.zD)
        req_kwargs = {
            'x0_mean': np.atleast_1d(x0_mean),
            'x0_cov': np.atleast_2d(x0_cov),
            'q_mean': np.zeros(self.qD),
            'q_cov': 0.01 * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.atleast_2d(r_cov),
            'q_gain': np.eye(2)
        }
        super(PendulumGaussSSM, self).__init__(**req_kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] + x[1] * self.dt, x[1] - self.g * self.dt * np.sin(x[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.array([np.sin(x[0])]) + r

    def par_fcn(self, time):
        pass  # pendulum model does not have time-varying parameters

    def dyn_fcn_dx(self, x, q, pars):
        return np.array([[1.0, self.dt],
                         [-self.g * self.dt * np.cos(x[0]), 1.0]])

    def meas_fcn_dx(self, x, r, pars):
        return np.array([np.cos(x[0]), 0.0])


class CoordinatedTurnBearingsOnlyTrackingGaussSSM(GaussianStateSpaceModel):
    """
    Bearings only target tracking in 2D using multiple sensors as in [3]_.

    TODO:
    Coordinated turn model [1]_ assumes constant turn rate (not implemented).
    Model in [2]_ is implemented here, where the turn rate can change in time and measurements are range and bearing.
    [3]_ considers only bearing measurements.

    State
    -----
    x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position [m]
        v_1, v_2 - target velocity [m/s]
        omega - target turn rate [deg/s]

    Measurements
    ------------


    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    xD = 5
    zD = 4  # measurement dimension == # sensors
    qD = 5
    rD = 4  # measurement noise dimension == # sensors

    q_additive = True
    r_additive = True

    rho_1, rho_2 = 0.1, 1.75e-4  # noise intensities

    def __init__(self, dt=0.1, sensor_pos=np.vstack((np.eye(2), -np.eye(2)))):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        sensor_pos :
            sensor [x, y] positions in rows
        """
        self.dt = dt
        self.sensor_pos = sensor_pos  # np.vstack((np.eye(2), -np.eye(2)))
        kwargs = {
            'x0_mean': np.array([1000, 300, 1000, 0, -3.0 * np.pi / 180]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([100, 10, 100, 10, 0.1]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array(
                [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
                 [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
                 [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
                 [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
                 [0, 0, 0, 0, self.rho_2 * self.dt]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': 10e-3 * np.eye(self.rD)  # 10e-3 rad == 10 mrad
        }
        super(CoordinatedTurnBearingsOnlyTrackingGaussSSM, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        """
        Model describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).

        Parameters
        ----------
        x
        q
        args


        Returns
        -------

        """
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om
        d = (1 - np.cos(om * self.dt)) / om
        mdyn = np.array([[1, c, 0, -d, 0],
                         [0, b, 0, -a, 0],
                         [0, d, 1, c, 0],
                         [0, a, 0, b, 0],
                         [0, 0, 0, 0, 1]])
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        """
        Bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x
        r
        args

        Returns
        -------

        """
        a = x[2] - self.sensor_pos[:, 1]
        b = x[0] - self.sensor_pos[:, 0]
        return np.arctan(a / b) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


class CoordinatedTurnRadarGaussSSM(GaussianStateSpaceModel):
    """
    Tracking of a maneuvering target in 2D using radar measurements.

    Parameters
    ----------
    dt : float
        Time interval between two consecutive measurements in seconds.

    sensor_pos : (num_sensors, 2) ndarray
        Array containing sensor [x, y] positions in rows.

    Notes
    -----
    Coordinated turn model [1]_ assumes constant turn rate (not implemented). Model in [2]_ is implemented here, where
    the turn rate can change in time and measurements are range and bearing. [3]_ considers only bearing measurements.

    State: :math:`\\mathbf{x} = [x, \dot{x}, y, \dot{y}, \\omega]`, where

        :math`x`, :math:`y`
            target position in 2D [m]
        :math`\\dot{x}`, :math`\\dot{y}`
            target velocity [m/s]
        :math:`\\omega`
            target turn rate [deg/s]

    Measurements: there are `num_sensors` of bearing measurements, given by
    .. math::
    \[
        \\theta_s = \\mathrm{atan2}\\left( \\frac{y - y_s}{x - x_s} \\right)
    \]


    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    xD = 5
    zD = 4  # measurement dimension == # sensors
    qD = 5
    rD = 4  # measurement noise dimension == # sensors

    q_additive = True
    r_additive = True

    rho_1, rho_2 = 0.5, 1e-6  # noise intensities

    def __init__(self, dt=0.2, sensor_pos=np.vstack((np.eye(2), -np.eye(2)))):
        self.dt = dt
        self.sensor_pos = sensor_pos  # np.vstack((np.eye(2), -np.eye(2)))
        kwargs = {
            'x0_mean': np.array([130, 25, -20, 1, -4 * np.pi / 180]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([5, 5, 2e4, 10, 1e-7]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array(
                [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
                 [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
                 [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
                 [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
                 [0, 0, 0, 0, self.rho_2 * self.dt]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': 1e-2 * np.eye(self.rD)  # 1e-2 rad == 10 mrad
        }
        super(CoordinatedTurnRadarGaussSSM, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        """
        Equation describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        q : (dim_q, ) ndarray
            State noise vector.

        args : tuple
            Unused.

        Returns
        -------
        : (dim_x, ) ndarray
            System state in the next time step.
        """
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om
        d = (1 - np.cos(om * self.dt)) / om
        mdyn = np.array([[1, c, 0, -d, 0],
                         [0, b, 0, -a, 0],
                         [0, d, 1, c, 0],
                         [0, a, 0, b, 0],
                         [0, 0, 0, 0, 1]])
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        """
        Bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        r : (dim_r, ) ndarray
            Measurement noise vector.

        args : tuple
            Unused.

        Returns
        -------
        : (dim_y, ) ndarray
            Bearing measurements provided by each sensor.
        """
        a = x[2] - self.sensor_pos[:, 1]
        b = x[0] - self.sensor_pos[:, 0]
        return np.arctan(a / b) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


class ReentryVehicleRadarTrackingGaussSSM(GaussianStateSpaceModel):
    """
    Radar tracking of the reentry vehicle entering Earth's atmosphere as described in [1]_.

    Parameters
    ----------
    dt : float
        Time interval between two consecutive measurements.

    Notes
    -----
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
    b0 = -0.59783  # ballistic coefficient of a typical vehicle
    sx, sy = R0, 0  # radar location

    def __init__(self, dt=0.1):
        self.dt = dt
        kwargs = {
            'x0_mean': np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932]),  # m, m/s, m m/s, rad/s
            'x0_cov': np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1]),  # m^2, m^2/s^2, m^2, m^2/s^2, rad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': self.dt ** -1 * np.array(
                [[2.4064e-5, 0, 0],
                 [0, 2.4064e-5, 0],
                 [0, 0, 1e-6]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[1e-6, 0],
                               [0, 0.17e-3 ** 2]]),
            'q_gain': np.vstack((np.zeros((2, 3)), np.eye(3)))
        }
        super(ReentryVehicleRadarTrackingGaussSSM, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        """
        Equation describing dynamics of the reentry vehicle.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        q : (dim_q, ) ndarray
            State noise vector.

        pars :
            Unused.

        Returns
        -------
        : (dim_x, ) ndarray
            System state in the next time step.
        """
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
        return np.array([x[0] + self.dt * x[2],
                         x[1] + self.dt * x[3],
                         x[2] + self.dt * (D * x[2] + G * x[0]) + q[0],
                         x[3] + self.dt * (D * x[3] + G * x[1]) + q[1],
                         x[4] + q[2]])

    def meas_fcn(self, x, r, pars):
        """
        Bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        r : (dim_r, ) ndarray
            Measurement noise vector.

        pars : tuple
            Unused.

        Returns
        -------
        : (dim_y, ) ndarray
            Range and bearing measurements.
        """
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


class ReentryVehicleRadarTrackingSimpleGaussSSM(GaussianStateSpaceModel):
    """
    Simplified model of the reentry vehicle entering Earth's atmosphere.

    Parameters
    ----------
    dt : float
        Time interval between two consecutive measurements.

    Notes
    -----
    The object moves only vertically and the radar is positioned at 30km above the ground and 30km away from the
    vertical path of the falling object.

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
    .. [1] Julier, S. J., & Uhlmann, J. K. (2004). Unscented Filtering and Nonlinear Estimation.
           Proceedings of the IEEE, 92(3), 401-422
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

    def __init__(self, dt=0.1):
        self.dt = dt
        kwargs = {
            'x0_mean': np.array([90, 6, 1.7]),  # km, km/s
            'x0_cov': np.diag([0.3048**2, 1.2192**2, 10]),  # km^2, km^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': np.array([[0.03048**2]]),
            'q_gain': np.vstack(np.eye(3))
        }
        super(ReentryVehicleRadarTrackingSimpleGaussSSM, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        """
        Discretized equation describing simplified dynamics of the reentry vehicle.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        q : (dim_q, ) ndarray
            State noise vector.

        pars :
            Unused.

        Returns
        -------
        : (dim_x, ) ndarray
            System state in the next time step.
        """
        return np.array([x[0] - self.dt * x[1] + q[0],
                         x[1] - self.dt * np.exp(-self.Gamma*x[0])*x[1]**2*x[2] + q[1],
                         x[2] + q[2]])

    def meas_fcn(self, x, r, pars):
        """
        Range (distance) to the target.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        r : (dim_r, ) ndarray
            Measurement noise vector.

        pars : tuple
            Unused.

        Returns
        -------
        : (dim_y, ) ndarray
            Range measurement.
        """
        # range
        rng = np.sqrt(self.sx ** 2 + (x[0] - self.sy) ** 2)
        return np.array([rng]) + r

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass


class UNGMGaussSSM(GaussianStateSpaceModel):
    """
    Univariate Non-linear Growth Model frequently used as a benchmark.
    """

    # CLASS VARIABLES are shared by all instances of this class
    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1
    q_additive = True
    r_additive = True

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        super(UNGMGaussSSM, self).__init__(**kwargs)
        self.set_pars('x0_mean', np.atleast_1d(x0_mean))
        self.set_pars('x0_cov', np.atleast_2d(x0_cov))
        self.set_pars('q_mean', np.atleast_1d(q_mean))
        self.set_pars('q_cov', np.atleast_2d(q_cov))
        self.set_pars('r_mean', np.atleast_1d(r_mean))
        self.set_pars('r_cov', np.atleast_2d(r_cov))
        self.set_pars('q_gain', np.eye(1))

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * pars[0])]) + q

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * x[0] ** 2]) + r

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * x[0]])


class UNGMNonAdditiveGaussSSM(GaussianStateSpaceModel):
    """
    Univariate Non-linear Growth Model with non-additive noise for testing.
    """

    # CLASS VARIABLES are shared by all instances of this class
    xD = 1  # state dimension
    zD = 1  # measurement dimension
    qD = 1
    rD = 1

    q_additive = False
    r_additive = False

    def __init__(self, x0_mean=0.0, x0_cov=1.0, q_mean=0.0, q_cov=10.0, r_mean=0.0, r_cov=1.0, **kwargs):
        super(UNGMNonAdditiveGaussSSM, self).__init__(**kwargs)
        self.set_pars('x0_mean', np.atleast_1d(x0_mean))
        self.set_pars('x0_cov', np.atleast_2d(x0_cov))
        self.set_pars('q_mean', np.atleast_1d(q_mean))
        self.set_pars('q_cov', np.atleast_2d(q_cov))
        self.set_pars('r_mean', np.atleast_1d(r_mean))
        self.set_pars('r_cov', np.atleast_2d(r_cov))

    def dyn_fcn(self, x, q, pars):
        return np.asarray([0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * q[0] * np.cos(1.2 * pars[0])])

    def meas_fcn(self, x, r, pars):
        return np.asarray([0.05 * r[0] * x[0] ** 2])

    def par_fcn(self, time):
        return np.atleast_1d(time)

    def dyn_fcn_dx(self, x, q, pars):
        return np.asarray([0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * pars[0])])

    def meas_fcn_dx(self, x, r, pars):
        return np.asarray([0.1 * r[0] * x[0], 0.05 * x[0] ** 2])