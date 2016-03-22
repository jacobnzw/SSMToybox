import numpy as np

from inference.ssinfer import StateSpaceInference
from models.ssmodel import StateSpaceModel
from transforms.bayesquad import GPQuadDer
from transforms.quad import Unscented


class GPQuadDerKalman(StateSpaceInference):
    """
    GP quadrature filter and smoother.
    """

    def __init__(self, sys, usp_dyn=None, usp_meas=None, hyp_dyn=None, hyp_meas=None, which_der=None):
        assert isinstance(sys, StateSpaceModel)
        if usp_dyn is not None and usp_meas is not None and hyp_dyn is not None and hyp_meas is not None:
            self.usp_dyn, self.usp_meas, self.hyp_dyn, self.hyp_meas = usp_dyn, usp_meas, hyp_dyn, hyp_meas
        else:
            self._set_default_usp_hyp(sys)
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        self.tf = GPQuadDer(nq, self.usp_dyn, self.hyp_dyn, which_der)
        self.th = GPQuadDer(nr, self.usp_meas, self.hyp_meas, which_der)
        super(GPQuadDerKalman, self).__init__(self.tf, self.th, sys)

    def _set_default_usp_hyp(self, sys):
        nq = sys.xD if sys.q_additive else sys.xD + sys.qD
        nr = sys.xD if sys.r_additive else sys.xD + sys.rD
        kappa, alpha, beta = 0.0, 1.0, 2.0
        self.usp_dyn = Unscented.unit_sigma_points(nq, kappa, alpha, beta)
        self.usp_meas = Unscented.unit_sigma_points(nr, kappa, alpha, beta)
        # self.hyp_dyn = {'sig_var': 1.1, 'lengthscale': 3.0 * np.ones((nq,)), 'noise_var': 1e-8}
        # self.hyp_meas = {'sig_var': 1.1, 'lengthscale': 3.0 * np.ones((nr,)), 'noise_var': 1e-8}
        self.hyp_dyn = {'bias': 1.1, 'variance': 0.3 * np.ones((nq,)), 'noise_var': 1e-8}
        self.hyp_meas = {'bias': 1.1, 'variance': 0.3 * np.ones((nr,)), 'noise_var': 1e-8}


def main():
    from models.ungm import ungm_filter_demo
    central_sigma = np.array([0])
    # ungm_filter_demo(GPQuadDerKalman, which_der=central_sigma)
    from models.pendulum import pendulum_filter_demo
    hyp = {'bias': 1.1, 'variance': 0.01 * np.ones((2,)), 'noise_var': 1e-8}
    usp = np.zeros((2, 1))  # central sigma, GPQuadDerKalman ~= EKF)
    pendulum_filter_demo(GPQuadDerKalman,
                         usp_dyn=usp, usp_meas=usp,
                         hyp_dyn=hyp, hyp_meas=hyp,
                         which_der=central_sigma)


if __name__ == '__main__':
    main()
