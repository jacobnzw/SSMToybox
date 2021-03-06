from abc import ABCMeta, abstractmethod

import numpy as np

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

    noise_gain : (dim_state, dim_noise) ndarray, optional
        Noise gain matrix.

    Attributes
    ----------
    dim_state : int
        Input dimensionality of the state transition function.

    dim_noise : int
        Dimensionality of the process noise vector.

    noise_additive : bool
        Indicates additivity of the noise. `True` if noise is additive, `False` otherwise.
    """

    dim_in = None
    dim_state = None
    dim_noise = None
    noise_additive = None

    def __init__(self, init_rv, noise_rv, noise_gain=None):
        # input dimensionality of the dynamics function depends on noise additivity
        self.dim_in = self.dim_state if self.noise_additive else self.dim_state + self.dim_noise
        # distribution of initial conditions
        self.init_rv = init_rv
        # distribution of process noise
        self.noise_rv = noise_rv
        # zero vec for convenience
        self.zero_q = np.zeros(self.dim_noise)  # TODO rename to q_zero
        if noise_gain is None:
            noise_gain = np.eye(self.dim_state, self.dim_noise)
        self.noise_gain = noise_gain

    @abstractmethod
    def dyn_fcn(self, x, q, time):
        """Discrete-time system dynamics.

        Abstract method for the discrete-time system dynamics.

        Parameters
        ----------
        x : (dim_state, ) ndarray
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
        x : (dim_state, ) ndarray
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
    def dyn_fcn_dx(self, x, q, time):
        """Jacobian of the system dynamics.

        Abstract method for the Jacobian of system dynamics. Jacobian is a matrix of first partial derivatives.

        Parameters
        ----------
        x : (dim_state, ) ndarray
            System state.

        q : (dim_noise, ) ndarray
            System noise.

        time : int
            Time index.

        Returns
        -------
        : (dim_state, dim_in) ndarray
            Jacobian matrix of the system dynamics. Note that in non-additive noise case
            `dim_in = dim_state + dim_noise`.
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
            assert len(xq) == self.dim_state
            if dx:
                out = self.dyn_fcn_dx(xq, self.zero_q, time)
            else:
                out = self.dyn_fcn(xq, self.zero_q, time)
        else:
            assert len(xq) == self.dim_state + self.dim_noise
            x, q = xq[:self.dim_state], xq[-self.dim_noise:]
            if dx:
                out = self.dyn_fcn_dx(x, q, time)
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
        x = np.zeros((self.dim_state, steps, mc_sims))
        # generate initial conditions, store initial states at k=0
        x[:, 0, :] = self.init_rv.sample(mc_sims)  # (D, mc_sims)

        # generate state and measurement noise
        q = self.noise_rv.sample((steps, mc_sims))

        # simulate SSM `mc_sims` times for `steps` time steps
        for imc in range(mc_sims):
            for k in range(1, steps):
                x[:, k, imc] = self.dyn_fcn(x[:, k-1, imc], q[:, k-1, imc], k-1)
        return x

    def simulate_continuous(self, duration, dt=0.1, mc_sims=1):
        """
        Computes continuous-time system state trajectory using the Euler-Maruyama SDE integration method.

        .. math::

            x_{k+1} = x_k + f(x_k, k) \\mathrm{d}t + q_k, \\quad q_k \\sim N(0, \\mathrm{d}t \\cdot Q)

        Parameters
        ----------
        duration : int
            Length of trajectory in seconds.

        dt : float, optional
            Discretization step.

        mc_sims : int, optional
            Number of Monte Carlo simulations.

        Returns
        -------
        : (dim_x, num_time_steps, num_mc_sims) ndarray
            State trajectories of the continuous-time dynamic system.
        """

        # ensure sensible values of dt
        if dt > duration:
            ValueError('Discretization step dt should be smaller than duration.')

        # allocate space for system state and noise
        steps = int(np.floor(duration / dt))
        x = np.zeros((self.dim_state, steps+1, mc_sims))
        # sample initial conditions and process noise
        x[:, 0, :] = self.init_rv.sample(mc_sims)  # (D, mc_sims)
        # Euler-Maruyama: noise must be sampled with variance V[q_k] = dt*Q
        q = (np.sqrt(dt)/dt) * self.noise_rv.sample((steps+1, mc_sims))

        # continuous-time system simulation
        for imc in range(mc_sims):
            for k in range(1, steps+1):
                # TODO: what about non-additive noise?
                # computes next state x(t + dt) by SDE integration
                x[:, k, imc] = x[:, k-1, imc] + dt * self.dyn_fcn_cont(x[:, k-1, imc], q[:, k-1, imc], k-1)
        return x[:, 1:, :]


class UNGMTransition(TransitionModel):
    """
    Univariate Nonlinear Growth Model (UNGM) with additive noise.

    Notes
    -----
    The model is

    .. math::
        x_{k+1} = 0.5 x_k + \\frac{25 x_k}{1 + x_k^2} + 8\\cos(1.2 k) + q_k

    Typically used with :math:`x_0 \\sim N(0, 1)`, :math:`q_k \\sim N(0, 10)`.
    """

    dim_state = 1
    dim_noise = 1
    noise_additive = True

    def __init__(self, init_rv, noise_rv):
        super(UNGMTransition, self).__init__(init_rv, noise_rv)

    def dyn_fcn(self, x, q, time):
        return np.asarray(0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * np.cos(1.2 * time)) + q

    def dyn_fcn_dx(self, x, q, time):
        return np.asarray([[0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2]])

    def dyn_fcn_cont(self, x, q, time):
        pass


class UNGMNATransition(TransitionModel):
    """
    Univariate Nonlinear Growth Model (UNGM) with non-additive noise.

    Notes
    -----
    The model is

    .. math::
        x_{k+1} = 0.5 x_k \\frac{25 x_k}{1 + x_k^2} + 8 q_k \\cos(1.2 k)

    Typically used with :math:`x_0 \\sim N(0, 1)`, :math:`q_k \\sim N(0, 10)`.
    """

    dim_state = 1
    dim_noise = 1
    noise_additive = False

    def __init__(self, init_rv, noise_rv):
        super(UNGMNATransition, self).__init__(init_rv, noise_rv)

    def dyn_fcn(self, x, q, time):
        return np.asarray(0.5 * x[0] + 25 * (x[0] / (1 + x[0] ** 2)) + 8 * q[0] * np.cos(1.2 * time))

    def dyn_fcn_cont(self, x, q, time):
        pass

    def dyn_fcn_dx(self, x, q, time):
        return np.asarray([[0.5 + 25 * (1 - x[0] ** 2) / (1 + x[0] ** 2) ** 2, 8 * np.cos(1.2 * time)]])


class Pendulum2DTransition(TransitionModel):
    """
    Pendulum with unit length and mass in 2D [1]_ (Example 5.1).

    Notes
    -----

    .. math ::

        \\begin{bmatrix}
            \\alpha_{k+1} \\\\
            \\dot{\\alpha}_{k+1}
        \\end{bmatrix} =
        \\begin{bmatrix}
            \\alpha_k + \\dot{\\alpha}_k\\Delta t \\\\
            \\dot{\\alpha}_k - g\\sin(\\alpha_k)\\Delta t
        \\end{bmatrix} + \\mathbf{q}_k

    where the state consists of

    :math:`\\alpha` angle from the perpendicular and the direction of the pendulum

    :math:`\\dot{\\alpha}` angular speed

    Reasonable default statistics: :math:`x_0 \\sim N(m_0, P_0),\\ q_k \\sim N(0, Q)` where

    .. math::
        m_0 = \\begin{bmatrix}1.5 \\\\ 0 \\end{bmatrix},\\quad P_0 = 0.01 I,\\quad Q = q_c
        \\begin{bmatrix}
            {\\Delta t}^3/3 & {\\Delta t}^2/2 \\\\
            {\\Delta t}^2/2 & {\\Delta t}
        \\end{bmatrix}

    References
    ----------
    .. [1] Sarkka, S., Bayesian Filtering and Smoothing, Cambridge University Press, 2013.
    """

    dim_state = 2
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

    :math:`\\mathbf{x} = [y, \\dot{y}, \\omega]`, where
        :math:`y`
            Altitude.

        :math:`\\dot{y}`
            Velocity.

        :math:`\\omega`
            (constant) ballistic coefficient.

    Notes
    -----
    .. math::
        \\begin{bmatrix}
            y_{k+1} \\\\
            \\dot{y}_{k+1} \\\\
            \\omega_{k+1}
        \\end{bmatrix} =
        \\begin{bmatrix}
            y_k - \\Delta t \\dot{y}_k \\\\
            \\dot{y}_k - \\Delta t \\exp(-\\gamma y_k) \\dot{y}_k^2 \\omega_k \\\\
            \\omega_k
        \\end{bmatrix} + q_k

    Reasonable default statistics: :math:`x_0 \\sim N(m_0, P_0),\\ q_k \\sim N(0, Q)`

    .. math::
        m_0 =
        \\begin{bmatrix}
            90 km \\\\ 6 km/s \\\\ 1.7
        \\end{bmatrix},\\ P_0 = \\mathrm{diag}([0.3048^2, 1.2192^2, 10]),\\ Q = 0


    References
    ----------
    .. [1] Julier, S. and Uhlmann, J., A General Method for Approximating Nonlinear Transformations of Probability
           Distributions, 1996
    """

    dim_state = 3
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

    "This type of problem has been identified by a number of authors [2]_ - [5]_ as being particularly stressful for
    filters and trackers because of the strong nonlinearities exhibited by the forces which act on the vehicle. There
    are three types of forces in effect. The most dominant is aerodynamic drag, which is a function of vehicle speed
    and has a substantial nonlinear variation in altitude. The second type of force is gravity, which accelerates the
    vehicle toward the center of the earth. The final forces are random buffeting terms." [1]_

    "The tracking problem is made more difficult by the fact that the drag properties of the vehicle might be only very
    crudely known." [1]_

    The model is specified in Cartesian `geocentric coordinates <https://en.wikipedia.org/wiki/ECEF>`_.

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

    .. math::
        \\mathbf{x}_{k+1} = \\mathbf{x}_k +
        \\begin{bmatrix}
            x_k\\Delta t \\\\
            y_k\\Delta t \\\\
            (D_k \\dot{x}_k + G_k x_k)\\Delta t \\\\
            (D_k \\dot{y}_k + G_k y_k)\\Delta t \\\\
            0
        \\end{bmatrix} + \\mathbf{G}\\mathbf{q}_k

    where the noise gain is

    .. math::
        \\mathbf{G} =
        \\begin{bmatrix}
            \\mathbf{0}_{2 \\times 3} \\\\
            \\mathbf{I}_{3 \\times 3}
        \\end{bmatrix}

    Reasonable default statistics:
    :math:`\\mathbf{x}_0 \\sim N(\\mathbf{m}_0, \\mathbf{P}_0),\\quad \\mathbf{q}_k \\sim N(0, \\mathbf{Q})`

    .. math::
        \\mathbf{m}_0 =
        \\begin{bmatrix}
            6500.4 km, 349.14 km, -1.8093 km/s, -6.7967 km/s, 0.6932
        \\end{bmatrix},

    .. math::
        \\mathbf{P}_0 = \\mathrm{diag}([10^{-6}, 10^{-6}, 10^{-6}, 10^{-6}, 1])

    Covariance of the Euler-Maruyama discretized model is

    .. math::
        \\mathbf{Q} = {\\Delta t}^{-1} \\mathrm{diag}([2.4064 \\times 10^{-5}\\ 2.4064\\times 10^{-5}\\ 10^{-6}])

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

    dim_state = 5
    dim_noise = 3
    noise_additive = True

    def __init__(self, init_rv, noise_rv, dt=0.1):
        self.dt = dt
        self.R0 = 6374  # Earth's radius
        self.H0 = 13.406
        self.Gm0 = 3.9860e5
        self.b0 = -0.59783  # ballistic coefficient of a typical vehicle
        noise_gain = np.vstack((np.zeros((2, self.dim_noise)), np.eye(self.dim_noise)))
        super(ReentryVehicle2DTransition, self).__init__(init_rv, noise_rv, noise_gain)

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


class CoordinatedTurnTransition(TransitionModel):
    """
    Coordinated turn model [1]_ assumes constant turn rate (not implemented).
    Model in [2]_ is implemented here, where the turn rate can change in time.
    [3]_ considers only bearing measurements.

    .. math::

        \\begin{bmatrix}
            x_{k+1} \\\\
            \\dot{x}_{k+1} \\\\
            y_{k+1} \\\\
            \\dot{y}_{k+1} \\\\
            \\omega_{k+1}
        \\end{bmatrix} =
        \\begin{bmatrix}
            1 & c & 0 & -d & 0 \\\\
            0 & b & 0 & -a & 0 \\\\
            0 & d & 1 &  c & 0 \\\\
            0 & a & 0 &  b & 0 \\\\
            0 & 0 & 0 &  0 & 1
        \\end{bmatrix} x_k + q_k

    where the state consists of

    :math:`x,\\ y` - target position [m]

    :math:`\\dot{x},\\ \\dot{y}` - target velocity [m/s]

    :math:`\\omega` - target turn rate [deg/s]

    and :math:`a = \\sin(\\omega \\Delta t)`, :math:`b = \\cos(\\omega \\Delta t)`,
    :math:`c = \\sin(\\omega \Delta t) / \\omega`, :math:`d = (1 - \\cos(\\omega \\Delta t)) / \\omega`.

    Reasonable default statistics [2]_: :math:`x_0 \\sim N(m_0, P_0),\\ q_k \\sim N(0, Q)` where

    .. math::

        m_0 =
        \\begin{bmatrix}
            1000 m \\\\
            300 m/s \\\\
            1000 m \\\\
            0 m/s \\\\
            -3.0 * \\pi / 180 rad/s
        \\end{bmatrix},\\ P_0 = \\mathrm{diag}([100,\\ 10,\\ 100,\\ 10,\\ 0.1])

    .. math::

        Q = \\mathrm{blkdiag}([\\rho_1 A,\\ \\rho_1 A,\\ \\rho_2\\Delta t])

    where noise intensities :math:`\\rho_1 = 0.1,\\ \\rho_2 = 1.75\\times 10^{-4}` and

    .. math::

        A =
        \\begin{bmatrix}
            {\\Delta t}^3/3 & {\\Delta t}^2/2 \\\\
            {\\Delta t}^2/2 & \\Delta t
        \\end{bmatrix}

    For more see [2]_ and [3]_.

    References
    ----------
    .. [1] Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001).
           Estimation with applications to tracking and navigation. Wiley-Blackwell.
    .. [2] Arasaratnam, I., and Haykin, S. (2009). Cubature Kalman Filters.
           IEEE Transactions on Automatic Control, 54(6), 1254-1269.
    .. [3] Sarkka, S., Hartikainen, J., Svensson, L., & Sandblom, F. (2015).
           On the relation between Gaussian process quadratures and sigma-point methods.
    """

    dim_state = 5
    dim_noise = 5

    noise_additive = True

    def __init__(self, init_rv, noise_rv, dt=0.1):
        """
        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
        super(CoordinatedTurnTransition, self).__init__(init_rv, noise_rv)
        self.dt = dt

    def dyn_fcn(self, x, q, *args):
        """
        Model describing an object in 2D plane moving with constant speed (magnitude of the velocity vector) and
        turning with a constant angular rate (executing a coordinated turn).
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

    def dyn_fcn_cont(self, x, q, time):
        pass

    def dyn_fcn_dx(self, x, r, time):
        pass


class ConstantTurnRateSpeed(TransitionModel):
    """
    Constant Turn-Rate and Speed (velocity magnitude).


    State vector: :math:`[p_x,\\ p_y,\\ v,\\ \\psi,\\ \\dot{\\psi}]`

    Noise vector: :math:`[\\nu_a,\\ \\nu_{\\ddot{\\psi}}]`

    Process model:

    For :math:`\\psi_k \\neq 0`:

    .. math::

        x_{k+1} = x_k +
        \\begin{bmatrix}
            \\frac{v_k}{\\dot{\\psi}_k} (\\sin(\\psi_k + \\dot{\\psi}_k \\Delta t) - \\sin(\\psi_k)) \\\\
            \\frac{v_k}{\\dot{\\psi}_k} (-\\cos(\\psi_k + \\dot{\\psi}_k \\Delta t) + \\cos(\\psi_k)) \\\\
            0 \\\\
            \\dot{\\psi}_k \\Delta t \\\\
            0
        \\end{bmatrix} +
        \\begin{bmatrix}
             \\frac{1}{2} (\\Delta t)^2 \\cos(\\psi_k) \\nu_{a,k} \\\\
             \\frac{1}{2} (\\Delta t)^2 \\sin(\\psi_k) \\nu_{\\dot{\\psi},k} \\\\
             \\Delta t \\nu_{a,k} \\\\
             \\frac{1}{2}(\\Delta t)^2 \\nu_{\\ddot{\\psi},k} \\\\
             \\Delta t \\nu_{\\ddot{\\psi},k}
        \\end{bmatrix}

    For :math:`\\psi_k = 0`:

    .. math::
        x_{k+1} = x_k +
        \\begin{bmatrix}
            \\Delta t v_k \\cos(\\psi_k) \\\\
            \\Delta t v_k \\sin(\\psi_k) \\\\
            \\Delta t \\nu_{a,k} \\\\
            \\Delta t \\psi_k + \\frac{1}{2}(\\Delta t)^2 \\nu_{\\dot{\\psi},k} \\\\
            \\Delta t \\nu_{\\dot{\\psi},k}
        \\end{bmatrix}

    Reasonable default statistics: :math:`x_0 \\sim N(0, 0.1I) \\ q_k \\sim N(0, \\mathrm{diag}([0.1,\\ 0.1\\pi])`
    :math:`r_k \\sim N(0, \\mathrm{diag}([0.3,\\ 0.03]))`

    """

    dim_state = 5
    dim_noise = 2
    noise_additive = False

    def __init__(self, init_rv, noise_rv, dt=0.05):
        super(ConstantTurnRateSpeed, self).__init__(init_rv, noise_rv)
        self.dt = dt

    def dyn_fcn(self, x, q, time):
        if x[4] == 0:
            # zero yaw rate case
            f = np.array([
                self.dt * x[2] * np.cos(x[3]),
                self.dt * x[2] * np.sin(x[3]),
                self.dt * q[0],
                self.dt * x[3] + 0.5 * self.dt ** 2 * q[1],
                self.dt * q[1]
            ])
        else:
            c = x[2] / x[4]
            f = np.array([
                c * (np.sin(x[3] + x[4] * self.dt) - np.sin(x[3])) + 0.5 * self.dt ** 2 * np.cos(x[3]) * q[0],
                c * (-np.cos(x[3] + x[4] * self.dt) + np.cos(x[3])) + 0.5 * self.dt ** 2 * np.sin(x[3]) * q[0],
                self.dt * q[0],
                self.dt * x[3] + 0.5 * self.dt ** 2 * q[1],
                self.dt * q[1]
            ])
        return x + f

    def dyn_fcn_dx(self, x, q, time):
        pass

    def dyn_fcn_cont(self, x, q, time):
        return np.array([x[2]*np.cos(x[3]), x[2]*np.sin(x[3]), 0, x[4], 0])


class ConstantVelocity(TransitionModel):
    """
    Constant velocity (CT) model.

    State vector: :math:`[p_x, v_x, p_y, v_y]`
    Noise vector: :math:`[q_1, q_2]`

    Process equation:

    .. math::
        x_{k+1} =
        \\begin{bmatrix}
            1 & \\tau & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & \\tau \\\\
            0 & 0 & 0 & 1 \\\\
        \\end{bmatrix} x_k +
        \\begin{bmatrix}
            \\frac{\\tau^2}{2} & 0 \\\\
            \\tau & 0 \\\\
            0 & \\frac{\\tau^2}{2} \\\\
            0 & \\tau \\\\
        \\end{bmatrix} q_k

    Reasonable default statistics: :math:`x_0 \\sim N(m_0, P_0)` :math:`q_k \\sim N(0, \\mathrm{diag}([50,\\ 5]))`
    where

    .. math::

        m_0 =
        \\begin{bmatrix}
            10000 & 300 & 1000 & -40
        \\end{bmatrix}

    .. math::

        P_0 =
        \\mathrm{diag}
        \\begin{bmatrix}
            10000 & 100 & 10000 & 100
        \\end{bmatrix}

    """

    dim_state = 4
    dim_noise = 2
    noise_additive = True

    def __init__(self, init_rv, noise_rv, dt=0.1):
        self.dt = dt
        noise_gain = np.array([[self.dt**2/2, 0],
                               [self.dt, 0],
                               [0, self.dt**2/2],
                               [0, self.dt]])
        super(ConstantVelocity, self).__init__(init_rv, noise_rv, noise_gain)

    def dyn_fcn(self, x, q, time):
        A = np.array([
            [1, self.dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.dt],
            [0, 0, 0, 1]
        ])
        return A.dot(x) + self.noise_gain.dot(q)

    def dyn_fcn_dx(self, x, q, time):
        return np.array([[1, self.dt, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, self.dt],
                         [0, 0, 0, 1]]).T

    def dyn_fcn_cont(self, x, q, time):
        pass


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

    state_index : ndarray or list
        List of indices into the state vector picking out the relevant states for the measurement model.

    Attributes
    ----------
    dim_substate : int
        Dimensionality of the states fed into the measurement function. Often times, not all states are used for
        computation of the measurements (e.g. bearings only tracking).

    dim_out : int
        Output dimensionality of the measurement function (dimensionality of the measurement).

    dim_noise : int
        Dimensionality of the process noise vector.

    noise_additive : bool
        Indicates additivity of the measurement noise. `True` if noise is additive, `False` otherwise.
    """

    dim_substate = None
    dim_out = None  # dimensionality of the measurement
    dim_noise = None
    noise_additive = None

    def __init__(self, noise_rv, dim_state, state_index):
        # distribution of process noise
        self.noise_rv = noise_rv
        # zero vec for convenience
        self.zero_r = np.zeros(self.dim_noise)  # TODO: rename to r_zero
        # state index must contain same # of elements as input dimensionality of the measurement model
        if state_index is not None and len(state_index) != self.dim_substate:
            ValueError("State index must contain same number of elements as input dimensionality of the measurement "
                       "model: len(state_index) != self.dim_substate:")
        self.state_index = state_index
        # dimensionality of the input to the measurement function depends on noise additivity
        self.dim_in = dim_state if self.noise_additive else dim_state + self.dim_noise
        # system state dimensionality
        self.dim_state = dim_state

    @abstractmethod
    def meas_fcn(self, x, r, time):
        """Measurement model.

        Abstract method for the measurement model.

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

        When noise is additive, the measurement can be written as :math:`z_k = h(x_k) + r_k`,
        the function returns evaluation of :math:`h(x_k)`. On the contrary, when the noise is non-additive
        :math:`z_k = h(x_k, r_k)`, the evaluation of :math:`h(x_k, r_k)`.

        Parameters
        ----------
        xr : ndarray
            System state (potentially) augmented with the measurement noise and thus shape is
            (dim_state + dim_noise) or (dim_state, ) depending on noise additivity.

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

        # pick indices of the state relevant for the measurement model
        if self.state_index is not None:
            xr = xr[self.state_index]

        if self.noise_additive:
            if dx:
                out = np.zeros((self.dim_out, self.dim_state))
                out[:, self.state_index] = self.meas_fcn_dx(xr, self.zero_r, time)
            else:
                out = self.meas_fcn(xr, self.zero_r, time)
        else:
            assert len(xr) == self.dim_substate + self.dim_noise
            x, r = xr[:self.dim_substate], xr[-self.dim_noise:]
            if dx:
                out = np.zeros((self.dim_out, self.dim_state + self.dim_noise))
                jac_out = self.meas_fcn_dx(x, r, time)
                out[:, self.state_index] = jac_out[:, :self.dim_substate]
                out[:, self.dim_state:] = jac_out[:, self.dim_substate:]
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

        # pick indices of the state relevant for the measurement model
        if self.state_index is not None:
            x = x[self.state_index]

        d, steps, mc_sims = x.shape

        # Generate measurement noise
        r = self.noise_rv.sample((steps, mc_sims))
        y = np.zeros((self.dim_out, steps, mc_sims))
        for imc in range(mc_sims):
            for k in range(steps):
                # measurement y[0] stored at index 0 happens at time 1
                y[:, k, imc] = self.meas_fcn(x[:, k, imc], r[:, k, imc], k+1)
        return y


class UNGMMeasurement(MeasurementModel):
    """
    Measurement model for the UNGM.

    .. math::
        z_k = 0.05 x_k^2 + r_k

    Reasonable default statistics: :math:`r_k \\sim N(0, 1)`
    """

    dim_substate = 1
    dim_out = 1
    dim_noise = 1
    noise_additive = True

    def __init__(self, noise_rv, dim_state, state_index=None):
        super(UNGMMeasurement, self).__init__(noise_rv, dim_state, state_index)

    def meas_fcn(self, x, r, time):
        return np.asarray([0.05 * x[0] ** 2]) + r

    def meas_fcn_dx(self, x, r, time):
        return np.asarray([0.1 * x[0]])


class UNGMNAMeasurement(MeasurementModel):
    """
    Measurement model for the UNGM with non-additive noise.

    .. math::
        z_k = 0.05 x_k^2 r_k

    Reasonable default statistics: :math:`r_k \\sim N(0, 1)`
    """

    dim_substate = 1
    dim_out = 1
    dim_noise = 1
    noise_additive = False

    def __init__(self, noise_rv, dim_state, state_index=None):
        super(UNGMNAMeasurement, self).__init__(noise_rv, dim_state, state_index)

    def meas_fcn(self, x, r, time):
        return np.asarray([0.05 * r[0] * x[0] ** 2])

    def meas_fcn_dx(self, x, r, time):
        return np.asarray([[0.1 * r[0] * x[0], 0.05 * x[0] ** 2]])


class Pendulum2DMeasurement(MeasurementModel):
    """
    Sine of an angle.

    Input
    -----
    .. math::
        z_k = \\sin(\\alpha_k) + r_k

    where :math:`\\alpha` is angle from the perpendicular line.

    Reasonable default statistics: :math:`r_k \\sim N(0, 0.1)`
    """

    dim_substate = 1
    dim_out = 1
    dim_noise = 1
    noise_additive = True

    def __init__(self, noise_rv, dim_state, state_index=None):
        super(Pendulum2DMeasurement, self).__init__(noise_rv, dim_state, state_index)

    def meas_fcn(self, x, r, time):
        return np.array([np.sin(x[0])]) + r

    def meas_fcn_dx(self, x, r, time):
        return np.array([[np.cos(x[0])]])


class RangeMeasurement(MeasurementModel):
    """
    Range measurement of vertically moving object for the Reentry1DTransition model.

    Notes
    -----
    .. math::
        z_k = \\sqrt{s_x^2 + (y_k - s_y)^2} + r_k

    where :math:`y` is the vertical coordinate and :math:`s_x,\\ s_y` are the sensor coordinates.

    Reasonable default statistics: :math:`r_k \\sim N(0, 0.03048^2)`

    """

    dim_substate = 1
    dim_out = 1
    dim_noise = 1
    noise_additive = True

    def __init__(self, noise_rv, dim_state, state_index=None):
        super(RangeMeasurement, self).__init__(noise_rv, dim_state, state_index)
        # radar location: 30km (~100k ft) above the surface, radar-to-body horizontal distance
        self.sx = 30
        self.sy = 30

    def meas_fcn(self, x, r, time):
        rng = np.sqrt(self.sx ** 2 + (x[0] - self.sy) ** 2)
        return np.array([rng]) + r

    def meas_fcn_dx(self, x, r, time):
        pass


class BearingMeasurement(MeasurementModel):
    """
    Bearing measurement.

    :math:`S` bearing measurements :math:`\\mathbf{z}_k = [z^{(1)}_k, \\ldots, z^{(S)}_k]` where

    .. math::
        z^{(s)}_k = \\mathrm{atan2}(y_k - p^{(s)}_y, x_k - p^{(s)}_x) + r^{(s)}_k

    Reasonable default statistics: :math:`r^{(s)}_k \\sim N(0, 10^{-3})`


    Parameters
    ----------
    sensor_pos : (num_sensors, 2) ndarray
        Positions of bearing sensors in 2D Cartesian plane.

    """

    dim_substate = 2
    dim_out = None
    dim_noise = None
    noise_additive = True

    def __init__(self, noise_rv, dim_state, state_index=None, sensor_pos=None):
        super(BearingMeasurement, self).__init__(noise_rv, dim_state, state_index)
        # default: 4 sensor positions
        if sensor_pos is None:
            sensor_pos = np.vstack((np.eye(2), -np.eye(2)))
        self.sensor_pos = sensor_pos
        # outputs == # sensors
        self.dim_out = len(self.sensor_pos)
        self.dim_noise = self.dim_out

    def meas_fcn(self, x, r, time):
        """
        Bearing measurement from the sensor to the moving object.
        """
        dx = x[0] - self.sensor_pos[:, 0]
        dy = x[1] - self.sensor_pos[:, 1]
        return np.arctan2(dy, dx) + r

    def meas_fcn_dx(self, x, r, time):
        pass


class Radar2DMeasurement(MeasurementModel):
    """
    Radar measurements.

    .. math::
        z_k =
        \\begin{bmatrix}
            \\sqrt{(x_k - s_x)^2 + (y_k - s_y)^2} \\\\
            \\mathrm{atan2}(y_k - s_y, x_k - s_x)
        \\end{bmatrix} + r_k

    Reasonable default statistics: :math:`r_k \\sim N(0, \\mathrm{diag}([10^{-6}\\ 0.17\\times 10^{-6}]))`
    """

    dim_substate = 2
    dim_out = 2
    dim_noise = 2
    noise_additive = True

    def __init__(self, noise_rv, dim_state, state_index=None, radar_loc=None):
        super(Radar2DMeasurement, self).__init__(noise_rv, dim_state, state_index)
        # set default radar location
        if radar_loc is None:
            radar_loc = np.array([0, 0])
        self.radar_loc = radar_loc

    def meas_fcn(self, x, r, time):
        """
        Range and bearing measurement from the sensor to the moving object.

        Parameters
        ----------
        x : (dim_x, ) ndarray
            State vector.

        r : (dim_r, ) ndarray
            Measurement noise vector.

        time : int
            Time index.

        Returns
        -------
        : (dim_y, ) ndarray
            Range and bearing measurements.
        """
        # TODO: (optionally) range rate measurements
        # range
        rng = np.sqrt((x[0] - self.radar_loc[0]) ** 2 + (x[1] - self.radar_loc[1]) ** 2)
        # bearing
        theta = np.arctan2((x[1] - self.radar_loc[1]), (x[0] - self.radar_loc[0]))
        return np.array([rng, theta]) + r

    def meas_fcn_dx(self, x, r, time):
        pass
