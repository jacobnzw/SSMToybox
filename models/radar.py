import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from ssmodel import *


class Reentry(StateSpaceModel):
    """
    Radar tracking of the reentry vehicle as described in [1]_.
    Vehicle is entering Earth's atmosphere at high altitude and with great speed, ground radar is tracking it.

    State
    -----
    position, velocity, aerodynamic parameter

    Measurements
    ------------
    range and bearing


    References
    ----------
    .. [1] Julier, S. J. and Uhlman, J. K. (2004)

    """

    def dyn_fcn(self, x, q, pars):
        pass

    def meas_fcn(self, x, r, pars):
        pass

    def par_fcn(self, time):
        pass

    def dyn_fcn_dx(self, x, q, pars):
        pass

    def meas_fcn_dx(self, x, r, pars):
        pass
