import unittest

import numpy as np
import matplotlib.pyplot as plt

from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement, UNGMNATransition, UNGMNAMeasurement, \
    Pendulum2DTransition, Pendulum2DMeasurement, ConstantTurnRateSpeed, Radar2DMeasurement, ReentryVehicle2DTransition
from ssmtoybox.utils import GaussRV


def default_bq_hypers(dyn, obs):
    hypers_f = np.atleast_2d(np.hstack((1, 3.0 * np.ones(dyn.dim_in))))
    hypers_h = np.atleast_2d(np.hstack((1, 3.0 * np.ones(obs.dim_in))))
    return hypers_f, hypers_h


class TestUNGM(unittest.TestCase):
    def test_dyn_fcn(self):
        pass

    def test_meas_fcn(self):
        pass

    def test_simulate(self):
        time_steps = 50
        # UNGM additive noise
        dim = 1
        init_dist = GaussRV(dim)
        noise_dist = GaussRV(dim, cov=np.atleast_2d(10.0))
        ungm_dyn = UNGMTransition(init_dist, noise_dist)
        ungm_meas = UNGMMeasurement(GaussRV(dim), ungm_dyn.dim_state)
        x = ungm_dyn.simulate_discrete(time_steps, mc_sims=20)
        y = ungm_meas.simulate_measurements(x)

        # UNGM non-additive noise
        ungmna_dyn = UNGMNATransition(init_dist, noise_dist)
        ungmna_meas = UNGMNAMeasurement(GaussRV(dim), ungm_dyn.dim_state)
        x = ungmna_dyn.simulate_discrete(time_steps, mc_sims=20)
        y = ungmna_meas.simulate_measurements(x)


class TestPendulum(unittest.TestCase):
    pass


class TestReentry(unittest.TestCase):

    def test_simulate_continuous(self):
        m0 = np.array([6500.4, 349.14, -1.8093, -6.7967, 0.6932])
        P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
        x0 = GaussRV(5, m0, P0)
        q = GaussRV(3, cov=np.diag([2.4064e-5, 2.4064e-5, 1e-6]))
        dyn = ReentryVehicle2DTransition(x0, q, dt=0.05)
        x = dyn.simulate_continuous(200, dt=0.05)

        plt.figure()
        plt.plot(x[0, ...], x[1, ...], color='r')
        plt.show()


class TestCTRS(unittest.TestCase):

    def test_simulate(self):
        # setup CTRS with radar measurements
        x0 = GaussRV(5, cov=0.1 * np.eye(5))
        q = GaussRV(2, cov=np.diag([0.1, 0.1 * np.pi]))
        r = GaussRV(2, cov=np.diag([0.3, 0.03]))
        dyn = ConstantTurnRateSpeed(x0, q)
        obs = Radar2DMeasurement(r, 5)
        x = dyn.simulate_discrete(100, 10)
        y = obs.simulate_measurements(x)

        plt.figure()
        plt.plot(x[0, ...], x[1, ...], alpha=0.25, color='b')
        plt.show()


class TestMeasurementModels(unittest.TestCase):

    def test_radar(self):
        r = GaussRV(2)
        dim_state = 5
        st_ind = np.array([0, 2])
        radar_location = np.array([6378.0, 0])
        obs = Radar2DMeasurement(r, dim_state, state_index=st_ind, radar_loc=radar_location)
        st, n = np.random.randn(5), np.random.randn(2)

        # check evaluation of the measurement function
        hx = obs.meas_eval(st, n, dx=False)
        self.assertTrue(hx.shape == (2, ))

        jac = obs.meas_eval(st, n, dx=True)
        # check proper dimensions
        self.assertEqual(jac.shape, (2, 5))
        # non-zero columns only at state_indexes
        self.assertTrue(np.array_equal(np.nonzero(jac.sum(axis=0))[0], st_ind))
