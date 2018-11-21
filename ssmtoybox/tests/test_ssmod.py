import unittest

import numpy as np

from ssmtoybox.ssinf import ExtendedKalman, CubatureKalman, UnscentedKalman, GaussHermiteKalman, GaussianProcessKalman, TPQKalman
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement, Pendulum2DTransition, Pendulum2DMeasurement, UNGMNonAdditiveGaussSSM


def default_bq_hypers(sys):
    nq = sys.xD if sys.q_additive else sys.xD + sys.qD
    nr = sys.xD if sys.r_additive else sys.xD + sys.rD
    hypers_f = np.atleast_2d(np.hstack((1, 3.0 * np.ones(nq))))
    hypers_h = np.atleast_2d(np.hstack((1, 3.0 * np.ones(nr))))
    return hypers_f, hypers_h


class TestUNGM(unittest.TestCase):
    # TODO: Tests for classes deriving from StateSpaceModel should ensure function implementation meet requirements.
    def test_dyn_fcn(self):
        pass

    def test_meas_fcn(self):
        pass

    def test_simulate_transition(self):
        ungm_dyn = UNGMTransition()
        ungmna_dyn = UNGMNATransition()
        ungm_dyn.simulate_discrete(50, mc_sims=20)
        ungmna_dyn.simulate_discrete(50, mc_sims=20)

    def test_simulate_measurement(self):
        ungm_meas = UNGMMeasurement()
        ungmna_meas = UNGMNAMeasurement()
        ungm_meas.simulate_measurements(50, mc_per_step=10)
        ungmna_meas.simulate_measurements(50, mc_per_step=10)

    def test_ungm_inference(self):
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with additive noise)
        """
        mod_dyn = UNGMTransition()
        mod_meas = UNGMMeasurement()
        x = mod_dyn.simulate_discrete(100, mc_sims=1)
        z = mod_meas.simulate_measurements(x)
        hyp_dyn, hyp_meas = default_bq_hypers(ssm)
        inf_method = (
            ExtendedKalman(ssm, mod_meas,,,
            UnscentedKalman(ssm, mod_meas,,,
            CubatureKalman(ssm, mod_meas,,,
            GaussHermiteKalman(ssm, mod_meas,,,
            GaussianProcessKalman(ssm, mod_meas, hyp_dyn, hyp_meas),
            TPQKalman(ssm, mod_meas, hyp_dyn, hyp_meas),
        )
        for inf in inf_method:
            inf.forward_pass(z[..., 0])
            inf.backward_pass()

    def test_ungm_nonadd_inference(self):
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with NON-additive noise)
        """
        ssm = UNGMNonAdditiveGaussSSM(x0_mean=0.1)
        x, z = ssm.simulate(100, mc_sims=1)
        hyp_dyn, hyp_meas = default_bq_hypers(ssm)
        inf_method = (
            ExtendedKalman(ssm, mod_meas,,,
            UnscentedKalman(ssm, mod_meas,,,
            CubatureKalman(ssm, mod_meas,,,
            GaussHermiteKalman(ssm, mod_meas,,,
            GaussianProcessKalman(ssm, mod_meas, hyp_dyn, hyp_meas),
            TPQKalman(ssm, mod_meas, hyp_dyn, hyp_meas),
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
        ssm = PendulumGaussSSM()
        x, z = ssm.simulate(100, mc_sims=1)
        hyp_dyn, hyp_meas = default_bq_hypers(ssm)
        inf_method = (
            ExtendedKalman(ssm, mod_meas,,,
            UnscentedKalman(ssm, mod_meas,,,
            CubatureKalman(ssm, mod_meas,,,
            GaussHermiteKalman(ssm, mod_meas,,,
            GaussianProcessKalman(ssm, mod_meas, hyp_dyn, hyp_meas),
            TPQKalman(ssm, mod_meas, hyp_dyn, hyp_meas),
        )
        for inf in inf_method:
            inf.forward_pass(z[..., 0])
            inf.backward_pass()
