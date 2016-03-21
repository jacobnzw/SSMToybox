# Evaluate performance of the following filters/smoothers
#   - EKF
#   - GPQKF w/ RBF kernel, one unit sigma-point (zero) and derivative observation
#   - GPQKF w/ AFFINE kernel, one unit sigma-point (zero) and derivative observation
#       * By using affine kernel we hope to recover an algorithm similar to EKF
#       * Difference to EKF is that this algorithm utilizes integral variance
#       * It could be construed as "Bayesian Quadrature EKF" (GP quadrature EKF)
#   - GPQKF w/ given kernel, UT unit sigma-points w/ derivative observation at the middle point ONLY
#       Kernels could be:
#       * RBF:
#       * AFFINE: This variant should be closest to the EKF
#       * HERMITE:

from inference import ExtendedKalman, GPQuadDerKalman
from models import UNGM

steps, mc = 50, 10  # time steps, mc simulations
# initialize SSM and generate some data
ssm = UNGM()
x, z = ssm.simulate()
# use only the central sigma-point
usp_0 = np.zeros(ssm.xD, 1)
# set the RBF kernel hyperparameters
hyp_rbf = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(ssm.xD, ), 'noise_var': 1e-8}
algorithms = (
    ExtendedKalman(ssm),
    GPQuadDerKalman(ssm, usp_dyn=usp_0, usp_meas=usp_0, hyp_dyn=hyp_rbf, hyp_meas=hyp_rbf),
)

#
