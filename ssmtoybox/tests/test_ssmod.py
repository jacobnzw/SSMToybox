import unittest

import numpy as np

from ssmtoybox.ssinf import ExtendedKalman, CubatureKalman, UnscentedKalman, GaussHermiteKalman, GaussianProcessKalman, TPQKalman
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement, UNGMNATransition, UNGMNAMeasurement, \
    Pendulum2DTransition, Pendulum2DMeasurement


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
        # TODO implement Gaussian Random Variable
        # init_dist = GaussRV(mean=np.array([0.0]), cov=np.array([1.0]))
        # noise_dist = GaussRV(mean=np.array([0.0]), cov=np.array([1.0]))
        ungm_dyn = UNGMTransition()
        ungm_meas = UNGMMeasurement()
        x = ungm_dyn.simulate_discrete(time_steps, mc_sims=20)
        y = ungm_meas.simulate_measurements(x)

        # UNGM non-additive noise
        ungmna_dyn = UNGMNATransition()
        ungmna_meas = UNGMNAMeasurement()
        x = ungmna_dyn.simulate_discrete(time_steps, mc_sims=20)
        y = ungmna_meas.simulate_measurements(x)

    def test_ungm_inference(self):  # FIXME: these methods test the filters, not the SSMs; should be in test_ssinf.py
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with additive noise)
        """
        dyn = UNGMTransition()
        meas = UNGMMeasurement()
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

    def test_ungm_nonadd_inference(self):
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with NON-additive noise)
        """
        mod_dyn = UNGMNATransition()
        mod_meas = UNGMNAMeasurement()
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
        mod_dyn = Pendulum2DTransition()
        mod_meas = Pendulum2DMeasurement()
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
