import unittest

import numpy as np

from ssmtoybox.ssinf import ExtendedKalman, CubatureKalman, UnscentedKalman, GaussHermiteKalman, GaussianProcessKalman, TPQKalman
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement, UNGMNATransition, UNGMNAMeasurement, \
    Pendulum2DTransition, Pendulum2DMeasurement
from ssmtoybox.utils import GaussRV


def default_bq_hypers(mod_dyn, mod_meas):
    nq = mod_dyn.dim_in if mod_dyn.noise_additive else mod_dyn.dim_in + mod_dyn.dim_noise
    nr = mod_meas.dim_in if mod_meas.noise_additive else mod_meas.dim_in + mod_meas.dim_noise
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
        ungm_meas = UNGMMeasurement(GaussRV(dim))
        x = ungm_dyn.simulate_discrete(time_steps, mc_sims=20)
        y = ungm_meas.simulate_measurements(x)

        # UNGM non-additive noise
        ungmna_dyn = UNGMNATransition(init_dist, noise_dist)
        ungmna_meas = UNGMNAMeasurement(GaussRV(dim))
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
        meas = UNGMMeasurement(GaussRV(dim))
        x = dyn.simulate_discrete(100, mc_sims=1)
        z = meas.simulate_measurements(x)
        hyp_dyn, hyp_meas = default_bq_hypers(dyn, meas)
        inf_method = (
            ExtendedKalman(dyn, meas),
            UnscentedKalman(dyn, meas),
            CubatureKalman(dyn, meas),
            GaussHermiteKalman(dyn, meas),
            GaussianProcessKalman(dyn, meas, hyp_dyn, hyp_meas),
            TPQKalman(dyn, meas, hyp_dyn, hyp_meas),
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
        mod_dyn = UNGMNATransition(init_dist, noise_dist)
        mod_meas = UNGMNAMeasurement(GaussRV(dim))
        x = mod_dyn.simulate_discrete(100)
        z = mod_meas.simulate_measurements(x)
        hyp_dyn, hyp_meas = default_bq_hypers(mod_dyn, mod_meas)
        inf_method = (
            ExtendedKalman(mod_dyn, mod_meas),
            UnscentedKalman(mod_dyn, mod_meas),
            CubatureKalman(mod_dyn, mod_meas),
            GaussHermiteKalman(mod_dyn, mod_meas),
            GaussianProcessKalman(mod_dyn, mod_meas, hyp_dyn, hyp_meas),
            TPQKalman(mod_dyn, mod_meas, hyp_dyn, hyp_meas),
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
        mod_dyn = Pendulum2DTransition(x0, q, dt=dt)

        # measurement model: r = measurement noise RV
        r = GaussRV(1, cov=np.array([[0.1]]))
        mod_meas = Pendulum2DMeasurement(r)

        x = mod_dyn.simulate_discrete(100)
        z = mod_meas.simulate_measurements(x)
        hyp_dyn, hyp_meas = default_bq_hypers(mod_dyn, mod_meas)
        inf_method = (
            ExtendedKalman(mod_dyn, mod_meas),
            UnscentedKalman(mod_dyn, mod_meas),
            CubatureKalman(mod_dyn, mod_meas),
            GaussHermiteKalman(mod_dyn, mod_meas),
            GaussianProcessKalman(mod_dyn, mod_meas, hyp_dyn, hyp_meas),
            TPQKalman(mod_dyn, mod_meas, hyp_dyn, hyp_meas),
        )
        for inf in inf_method:
            inf.forward_pass(z[..., 0])
            inf.backward_pass()
