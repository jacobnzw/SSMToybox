import unittest

import numpy as np

from ssmtoybox.ssinf import ExtendedKalman, CubatureKalman, UnscentedKalman, GaussHermiteKalman, GaussianProcessKalman, TPQKalman
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement, UNGMNATransition, UNGMNAMeasurement, \
    Pendulum2DTransition, Pendulum2DMeasurement
from ssmtoybox.utils import GaussRV


def default_bq_hypers(dyn, obs):
    nq = dyn.dim_in if dyn.noise_additive else dyn.dim_in + dyn.dim_noise
    nr = dyn.dim_in if obs.noise_additive else dyn.dim_in + obs.dim_noise
    hypers_f = np.atleast_2d(np.hstack((1, 3.0 * np.ones(nq))))
    hypers_h = np.atleast_2d(np.hstack((1, 3.0 * np.ones(nr))))
    return hypers_f, hypers_h


class TestUNGM(unittest.TestCase):
    # TODO: Tests for classes deriving from StateSpaceModel should ensure function implementation meet requirements.
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
        ungm_meas = UNGMMeasurement(GaussRV(dim), ungm_dyn.dim_out)
        x = ungm_dyn.simulate_discrete(time_steps, mc_sims=20)
        y = ungm_meas.simulate_measurements(x)

        # UNGM non-additive noise
        ungmna_dyn = UNGMNATransition(init_dist, noise_dist)
        ungmna_meas = UNGMNAMeasurement(GaussRV(dim), ungm_dyn.dim_out)
        x = ungmna_dyn.simulate_discrete(time_steps, mc_sims=20)
        y = ungmna_meas.simulate_measurements(x)

    def test_ungm_inference(self):  # FIXME: these methods test the filters, not the SSMs; should be in test_ssinf.py
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with additive noise)
        """
        dim = 1
        init_dist = GaussRV(dim)
        noise_dist = GaussRV(dim, cov=np.atleast_2d(10.0))
        dyn = UNGMTransition(init_dist, noise_dist)
        obs = UNGMMeasurement(GaussRV(dim), dyn.dim_out)
        x = dyn.simulate_discrete(100, mc_sims=1)
        z = obs.simulate_measurements(x)
        hyp_dyn, hyp_meas = default_bq_hypers(dyn, obs)
        inf_method = (
            ExtendedKalman(dyn, obs),
            UnscentedKalman(dyn, obs),
            CubatureKalman(dyn, obs),
            GaussHermiteKalman(dyn, obs),
            GaussianProcessKalman(dyn, obs, hyp_dyn, hyp_meas),
            TPQKalman(dyn, obs, hyp_dyn, hyp_meas),
        )
        for inf in inf_method:
            inf.forward_pass(z[..., 0])
            inf.backward_pass()

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for inf in inf_method:
        #     plt.plot(inf.fi_mean[:, 1:].T, label=inf.__class__.__name__)
        # plt.plot(x[..., 0].T, label='True', color='r', ls='--')
        # plt.legend()
        # plt.show()

    def test_ungm_nonadd_inference(self):
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with NON-additive noise)
        """
        dim = 1
        init_dist = GaussRV(dim)
        noise_dist = GaussRV(dim, cov=np.atleast_2d(10.0))
        dyn = UNGMNATransition(init_dist, noise_dist)
        obs = UNGMNAMeasurement(GaussRV(dim), dyn.dim_out)
        x = dyn.simulate_discrete(100)
        z = obs.simulate_measurements(x)
        hyp_dyn, hyp_meas = default_bq_hypers(dyn, obs)
        inf_method = (
            ExtendedKalman(dyn, obs),
            UnscentedKalman(dyn, obs),
            CubatureKalman(dyn, obs),
            GaussHermiteKalman(dyn, obs),
            GaussianProcessKalman(dyn, obs, hyp_dyn, hyp_meas),
            TPQKalman(dyn, obs, hyp_dyn, hyp_meas),
        )
        for inf in inf_method:
            print(r"Testing {} ...".format(inf.__class__.__name__), end=' ')
            try:
                inf.forward_pass(z[..., 0])
                inf.backward_pass()
            except BaseException as e:
                print("Failed {}".format(e))
                continue
            print("OK")


class TestPendulum(unittest.TestCase):
    def test_pendulum_inference(self):
        """
        Test bunch of filters on a pendulum example
        """

        # transition model: x0 = initial state RV, q = state noise RV, dt = discretization period
        x0 = GaussRV(2, mean=np.array([1.5, 0]), cov=0.01 * np.eye(2))
        dt = 0.01
        q = GaussRV(2, cov=0.01 * np.array([[(dt ** 3) / 3, (dt ** 2) / 2], [(dt ** 2) / 2, dt]]))
        dyn = Pendulum2DTransition(x0, q, dt=dt)

        # measurement model: r = measurement noise RV
        r = GaussRV(1, cov=np.array([[0.1]]))
        obs = Pendulum2DMeasurement(r, dyn.dim_out)

        x = dyn.simulate_discrete(100)
        z = obs.simulate_measurements(x)
        hyp_dyn, hyp_meas = default_bq_hypers(dyn, obs)
        inf_method = (
            ExtendedKalman(dyn, obs),
            UnscentedKalman(dyn, obs),
            CubatureKalman(dyn, obs),
            GaussHermiteKalman(dyn, obs),
            GaussianProcessKalman(dyn, obs, hyp_dyn, hyp_meas),
            TPQKalman(dyn, obs, hyp_dyn, hyp_meas),
        )
        for inf in inf_method:
            inf.forward_pass(z[..., 0])
            inf.backward_pass()


class TestReentry(unittest.TestCase):
    pass