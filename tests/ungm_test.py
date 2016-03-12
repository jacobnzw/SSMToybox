import unittest

from inference import *
from models.ungm import UNGM, UNGMnonadd


class TestUNGM(unittest.TestCase):

    def test_dyn_fcn(self):
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
        inf_method = (
            ExtendedKalman(ssm),
            UnscentedKalman(ssm, kap=0.0),
            CubatureKalman(ssm),
            GaussHermiteKalman(ssm),
            GPQuadKalman(ssm),
            TPQuadKalman(ssm),
        )
        for inf in inf_method:
            inf.forward_pass(z[..., 0])
            inf.backward_pass()

    def test_ungm_nonadd_inference(self):
        """
        Test bunch of filters on Univariate Non-linear Growth Model (with NON-additive noise)
        """
        ssm = UNGMnonadd()
        x, z = ssm.simulate(100, mc_sims=1)
        inf_method = (
            ExtendedKalman(ssm),
            UnscentedKalman(ssm),
            CubatureKalman(ssm),
            GaussHermiteKalman(ssm),
            GPQuadKalman(ssm),
            TPQuadKalman(ssm),
        )
        for inf in inf_method:
            print r"Testing {} ...".format(inf.__class__.__name__),
            try:
                inf.forward_pass(z[..., 0])
                inf.backward_pass()
            except BaseException as e:
                print "Failed {}".format(e)
                continue
            print "OK"
