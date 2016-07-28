import unittest

import numpy as np

from inference.extended import ExtendedKalman
from inference.unscented import UnscentedKalman
from inference.cubature import CubatureKalman
from inference.gausshermite import GaussHermiteKalman
from inference.gpquad import GPQKalman
from inference.tpquad import TPQKalman
from models.pendulum import Pendulum
from models.ungm import UNGM, UNGMnonadd


def default_bq_hypers(sys):
    nq = sys.xD if sys.q_additive else sys.xD + sys.qD
    nr = sys.xD if sys.r_additive else sys.xD + sys.rD
    hypers_f = {'alpha': 1.0, 'el': 3.0 * np.ones((nq,))}
    hypers_h = {'alpha': 1.0, 'el': 3.0 * np.ones((nr,))}
    return hypers_f, hypers_h


class TestUNGM(unittest.TestCase):
    # TODO: Tests for classes deriving from StateSpaceModel should ensure function implementation meet requirements.
    def test_dyn_fcn(self):
        pass

    def test_meas_fcn(self):
        pass

    def test_simulate(self):
        ungm = UNGM()
        ungmna = UNGMnonadd()
        ungm.simulate(50, mc_sims=20)
        ungmna.simulate(50, mc_sims=20)

    def test_ungm_inference(self):
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with additive noise)
        """
        ssm = UNGM()
        x, z = ssm.simulate(100, mc_sims=1)
        hyp_dyn, hyp_meas = default_bq_hypers(ssm)
        inf_method = (
            ExtendedKalman(ssm),
            UnscentedKalman(ssm, kappa=0.0),
            CubatureKalman(ssm),
            GaussHermiteKalman(ssm),
            GPQKalman(ssm, 'rbf', 'ut', hyp_dyn, hyp_meas),
            TPQKalman(ssm, 'rbf', 'ut', hyp_dyn, hyp_meas),
        )
        for inf in inf_method:
            inf.forward_pass(z[..., 0])
            inf.backward_pass()

    def test_ungm_nonadd_inference(self):
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with NON-additive noise)
        """
        ssm = UNGMnonadd(x0_mean=0.1)
        x, z = ssm.simulate(100, mc_sims=1)
        hyp_dyn, hyp_meas = default_bq_hypers(ssm)
        inf_method = (
            ExtendedKalman(ssm),
            UnscentedKalman(ssm),
            CubatureKalman(ssm),
            GaussHermiteKalman(ssm),
            GPQKalman(ssm, 'rbf', 'ut', hyp_dyn, hyp_meas),
            TPQKalman(ssm, 'rbf', 'ut', hyp_dyn, hyp_meas),
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
        ssm = Pendulum()
        x, z = ssm.simulate(100, mc_sims=1)
        hyp_dyn, hyp_meas = default_bq_hypers(ssm)
        inf_method = (
            ExtendedKalman(ssm),
            UnscentedKalman(ssm),
            CubatureKalman(ssm),
            GaussHermiteKalman(ssm),
            GPQKalman(ssm, 'rbf', 'ut', hyp_dyn, hyp_meas),
            TPQKalman(ssm, 'rbf', 'ut', hyp_dyn, hyp_meas),
        )
        for inf in inf_method:
            inf.forward_pass(z[..., 0])
            inf.backward_pass()
