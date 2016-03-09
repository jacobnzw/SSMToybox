import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from ssmodel import *


class BearingsOnlyTracking(StateSpaceModel):
    """
    Bearings only target tracking in 2D using 4 sensors.

    State: x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position [m]
        v_1, v_2 - target velocity [m/s]
        omega - target turn rate [deg/s]
    """

    xD = 5
    zD = 4
    qD = 5
    rD = 4
    q_additive = True
    r_additive = True
    rho_1, rho_2 = 0.1, 1.75e-4  # noise intensities

    def __init__(self, dt=1.0, sensor_pos=np.vstack((np.eye(2), -np.eye(2)))):
        """

        :param dt: time interval between two consecutive measurements
        :param sensor_pos: sensor [x, y] positions in rows
        :return:
        """
        self.dt = dt
        self.sensor_pos = sensor_pos  # np.vstack((np.eye(2), -np.eye(2)))
        kwargs = {
            'x0_mean': np.array([1000, 300, 1000, 0, -3.0]),  # m, m/s, m m/s, deg/s
            'x0_cov': np.diag([100, 10, 100, 10, 0.100]),  # m^2, m^2/s^2, m^2, m^2/s^2, mrad^2/s^2
            'q_mean': np.zeros(self.qD),
            'q_cov': np.array(
                    [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
                     [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
                     [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
                     [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
                     [0, 0, 0, 0, self.rho_2 * self.dt]]),
            'r_mean': np.zeros(self.rD),
            'r_cov': 5.0 * np.eye(self.rD)
        }
        super(BearingsOnlyTracking, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
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
        a = x[2] - self.sensor_pos[:, 1]
        b = x[0] - self.sensor_pos[:, 0]
        return np.arctan(a / b) + r

    def par_fcn(self, time):
        pass


def bot_demo():
    steps = 100
    mc_simulations = 1
    ssm = BearingsOnlyTracking(dt=0.1)
    x, z = ssm.simulate(steps, mc_sims=mc_simulations)
    # plt.plot(x[0, ...], color='b', alpha=0.15, label='state trajectory')
    # plt.plot(z[0, ...], color='k', alpha=0.25, ls='None', marker='.', label='measurements')
    plt.figure()
    g = gridspec.GridSpec(4, 1)
    plt.subplot(g[:2, 0])
    for i in range(mc_simulations):
        plt.plot(x[0, :, i], x[2, :, i], alpha=0.85, color='b')
    plt.subplot(g[2, 0])
    plt.plot(x[0, :, 0])
    plt.subplot(g[3, 0])
    plt.plot(x[2, :, 0])
    plt.show()


if __name__ == '__main__':
    bot_demo()
