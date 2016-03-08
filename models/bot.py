from ssmodel import *


class BearingsOnlyTracking(StateSpaceModel):
    """
    Bearings only target tracking in 2D using 4 sensors.

    State: x = [x_1, v_1, x_2, v_2, omega]
        x_1, x_2 - target position []
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

    def __init__(self, dt=1, sensor_pos={}):
        self.dt = dt
        kwargs = {
            'q_cov': np.array(
                    [[self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0, 0, 0],
                     [self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0, 0, 0],
                     [0, 0, self.rho_1 * (self.dt ** 3) / 3, self.rho_1 * (self.dt ** 2) / 2, 0],
                     [0, 0, self.rho_1 * (self.dt ** 2) / 2, self.rho_1 * self.dt, 0],
                     [0, 0, 0, 0, self.rho_2 * self.dt]]
            ),
            'r_cov': 5.0 * np.eye(self.rD),

        }
        super(BearingsOnlyTracking, self).__init__(**kwargs)

    def dyn_fcn(self, x, q, *args):
        om = x[4]
        a = np.sin(om * self.dt)
        b = np.cos(om * self.dt)
        c = np.sin(om * self.dt) / om
        d = (1 - np.cos(om * self.dt)) / om
        mdyn = np.array(
                [[1, c, 0, -d, 0],
                 [0, b, 0, -a, 0],
                 [0, d, 1, c, 0],
                 [0, a, 0, b, 0],
                 [0, 0, 0, 0, 1]]
        )
        return mdyn.dot(x) + q

    def meas_fcn(self, x, r, *args):
        pass

    def par_fcn(self, time):
        pass
