import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ssinf import StateSpaceInference
from models.ssmodel import GaussianStateSpaceModel


class CoordinatedTurnBOT(GaussianStateSpaceModel):
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
        super(CoordinatedTurnBOT, self).__init__(**kwargs)

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


class CoordinatedTurnRadar(GaussianStateSpaceModel):
    """
    Maneuvering target tracking using radar measurements .

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

    rho_1, rho_2 = 0.5, 1e-6  # noise intensities

    def __init__(self, dt=0.2, sensor_pos=np.vstack((np.eye(2), -np.eye(2)))):
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
        super(CoordinatedTurnRadar, self).__init__(**kwargs)

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


class ReentryRadar(GaussianStateSpaceModel):
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
    b0 = -0.59783  # ballistic coefficient of a typical vehicle
    sx, sy = R0, 0  # radar location

    def __init__(self, dt=0.1):
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
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
        super(ReentryRadar, self).__init__(**kwargs)

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
        return np.array([x[0] + self.dt * x[2],
                         x[1] + self.dt * x[3],
                         x[2] + self.dt * (D * x[2] + G * x[0]) + q[0],
                         x[3] + self.dt * (D * x[3] + G * x[1]) + q[1],
                         x[4] + q[2]])

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


class ReentryRadarSimple(GaussianStateSpaceModel):
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
        """

        Parameters
        ----------
        dt :
            time interval between two consecutive measurements
        """
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
            'q_factor': np.vstack(np.eye(3))
        }
        super(ReentryRadarSimple, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, pars):
        return np.array([x[0] - self.dt * x[1] + q[0],
                         x[1] - self.dt * np.exp(-self.Gamma*x[0])*x[1]**2*x[2] + q[1],
                         x[2] + q[2]])

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


def bot_demo(steps=100, mc_sims=1):
    ssm = CoordinatedTurnBOT()
    x, z = ssm.simulate(steps, mc_sims=mc_sims)
    # plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.figure()
    g = gridspec.GridSpec(4, 1)
    plt.subplot(g[:2, 0])
    for i in range(mc_sims):
        plt.plot(x[0, :, i], x[2, :, i], alpha=0.35, color='b')
    plt.subplot(g[2, 0])
    plt.plot(x[0, :, :], 'b', alpha=0.25)
    plt.subplot(g[3, 0])
    plt.plot(x[2, :, :], 'b', alpha=0.25)
    plt.show()


def reentry_demo(steps=100, mc_sims=1):
    ssm = ReentryRadar()
    x, z = ssm.simulate(steps, mc_sims=mc_sims)
    # plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.figure()
    g = gridspec.GridSpec(2, 4)
    plt.subplot(g[:, :2])
    # Earth surface w/ radar position
    t = 0.02 * np.arange(-1, 4, 0.1)
    plt.plot(ssm.R0 * np.cos(t), ssm.R0 * np.sin(t), 'r', ssm.sx, ssm.sy, 'ko')
    # vehicle trajectory
    for i in range(mc_sims):
        plt.plot(x[0, :, i], x[1, :, i], alpha=0.35, color='b')
    plt.subplot(g[:, 2:], polar=True)
    plt.plot((z[1, :, 0]), z[0, :, 0], 'ko')
    plt.show()


def bot_filter_demo(filt_class, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    # sensor positions
    sen_pos = np.array([[1000, 900],
                        [600, 1000],
                        [1200, 800],
                        [1200, 1000]])
    # sen_pos = np.vstack((np.eye(2), -np.eye(2)))
    system = CoordinatedTurnBOT(sensor_pos=sen_pos)
    # create filter object, pass in additional kwargs
    filt = filt_class(system, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 130, 100
    x, z = system.simulate(time_steps, mc_sims=mc)
    print("Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc))
    rmse_filter = np.zeros((system.xD, mc))
    rmse_smoother = np.zeros((system.xD, mc))
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        rmse_filter[:, imc] = np.sqrt(np.mean((x[..., imc] - mean_f) ** 2, axis=1))
        rmse_smoother[:, imc] = np.sqrt(np.mean((x[..., imc] - mean_s) ** 2, axis=1))
        filt.reset()
    # print average filter/smoother RMSE
    print("Filter RMSE: {}".format((rmse_filter.mean(axis=1))))
    print("Smoother RMSE: {}".format((rmse_smoother.mean(axis=1))))
    # plot one realization of the system trajectory, measurements and filtered/smoothed state estimate
    plt.figure()
    time = list(range(1, time_steps))
    plt.plot(sen_pos[:, 0], sen_pos[:, 1], 'ko')
    plt.plot(x[0, :, imc], x[2, :, imc], color='r', ls='--', label='true state')
    # plt.plot(z[0, :, 0], color='k', ls='None', marker='o')
    plt.plot(mean_f[0, ...], mean_f[2, ...], color='b', label='filtered estimate')
    plt.plot(mean_s[0, ...], mean_s[2, ...], color='g', label='smoothed estimate')
    plt.legend()
    plt.show()


def reentry_filter_demo(filt_class, *args, **kwargs):
    assert issubclass(filt_class, StateSpaceInference)
    system = ReentryRadar()
    # create filter object, pass in additional kwargs
    filt = filt_class(system, *args, **kwargs)
    # simulate dynamic system for given number of steps and mc simulations
    time_steps, mc = 750, 50
    x, z = system.simulate(time_steps, mc_sims=mc)
    print("Running {} filter/smoother ({} time steps, {} MC simulations) ...".format(filt_class.__name__,
                                                                                     time_steps, mc))
    mse_filter = np.zeros((system.xD, time_steps, mc))
    mse_smoother = np.zeros((system.xD, time_steps, mc))
    for imc in range(mc):
        mean_f, cov_f = filt.forward_pass(z[..., imc])
        mean_s, cov_s = filt.backward_pass()
        mse_filter[..., imc] = (x[..., imc] - mean_f) ** 2
        mse_smoother[..., imc] = (x[..., imc] - mean_s) ** 2
        filt.reset()
    # position and velocity RMSE (time_steps, MC)
    rmse_filter_pos = np.sqrt(mse_filter[:2, ...].sum(axis=0))
    rmse_filter_vel = np.sqrt(mse_filter[2:4, ...].sum(axis=0))
    rmse_smoother_pos = np.sqrt(mse_smoother[:2, ...].sum(axis=0))
    rmse_smoother_vel = np.sqrt(mse_smoother[2:4, ...].sum(axis=0))
    # print average filter/smoother RMSE
    print("Filter stats:\n=============")
    print("Time-averaged RMSE (position): {}".format((rmse_filter_pos.mean())))
    print("Time-averaged RMSE (velocity): {}".format((rmse_filter_vel.mean())))
    print("Smoother stats:\n===============")
    print("Time-averaged RMSE (position): {}".format((rmse_smoother_pos.mean())))
    print("Time-averaged RMSE (velocity): {}".format((rmse_smoother_vel.mean())))
    # plot filter and smoother RMSE of position/velocity in time (MC averages)
    time = np.linspace(0, time_steps * system.dt, time_steps)
    g = gridspec.GridSpec(4, 2)
    plt.figure('Position and velocity RMSE (averaged over {} MC simulations)'.format(mc))
    ax = plt.subplot(g[1, 0])
    ax.set_title('Position: x[k]')
    ax.plot(time, mse_filter[0, ...].mean(axis=1))
    ax.plot(time, mse_smoother[0, ...].mean(axis=1))
    ax = plt.subplot(g[1, 1])
    ax.set_title('Position: y[k]')
    ax.plot(time, mse_filter[1, ...].mean(axis=1))
    ax.plot(time, mse_smoother[1, ...].mean(axis=1))
    ax = plt.subplot(g[0, :])
    ax.set_title('Position: \sqrt{x[k]^2 + y[k]^2}')
    ax.plot(time, rmse_filter_pos.mean(axis=1))
    ax.plot(time, rmse_smoother_pos.mean(axis=1))
    ax = plt.subplot(g[3, 0])
    ax.set_title('Velocity: \dot{x}[k]')
    ax.plot(time, mse_filter[2, ...].mean(axis=1))
    ax.plot(time, mse_smoother[2, ...].mean(axis=1))
    ax = plt.subplot(g[3, 1])
    ax.set_title('Velocity: \dot{y}[k]')
    ax.plot(time, mse_filter[3, ...].mean(axis=1))
    ax.plot(time, mse_smoother[3, ...].mean(axis=1))
    ax = plt.subplot(g[2, :])
    ax.set_title('Speed: \sqrt{\dot{x}[k]^2 + \dot{y}[k]^2}')
    ax.plot(time, rmse_filter_vel.mean(axis=1))
    ax.plot(time, rmse_smoother_vel.mean(axis=1))
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    reentry_demo(steps=750, mc_sims=100)
