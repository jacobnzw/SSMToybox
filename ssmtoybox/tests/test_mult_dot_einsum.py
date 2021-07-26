from unittest import TestCase

import logging
import numpy as np
import numpy.linalg as la
from scipy.linalg import cho_factor, cho_solve

from ssmtoybox.bq.bqmtran import GaussianProcessTransform
from ssmtoybox.mtran import MonteCarloTransform, UnscentedTransform
from ssmtoybox.ssmod import ReentryVehicle2DTransition
from ssmtoybox.utils import GaussRV

logging.basicConfig(level=logging.DEBUG)


def sym(a):
    return 0.5 * (a + a.T)


def cho_inv(a):
    n = a.shape[0]
    return cho_solve(cho_factor(a), np.eye(n))


class MultTest(TestCase):

    def test_dot_matvec_matmat(self):
        # Does numpy.dot use different subroutines for matrix/vector and matrix/matrix multiplication?

        # dot internals
        n, e = 300, 150
        A = 10 * np.random.randn(n, n)
        B = 50 * np.random.randn(e, n)
        A = sym(A)  # symmetrize A

        b = B[0, :]

        c = b.dot(A)
        C = B.dot(A)
        self.assertTrue(np.all(c == C[0, :]),
                        "MAX DIFF: {:.4e}".format(np.abs(c - C[0, :]).max()))

    def test_einsum_dot(self):
        # einsum and dot give different results?

        dim_in, dim_out = 2, 1
        rbf_kernel = {'name': 'rbf', 'params': np.hstack((np.ones((dim_out, 1)), 1 * np.ones((dim_out, dim_in))))}
        tf_mo = GaussianProcessTransform(dim_in, dim_out, rbf_kernel)
        iK, Q = tf_mo.model.iK, tf_mo.model.Q

        C1 = iK.dot(Q).dot(iK)
        C2 = np.einsum('ab, bc, cd', iK, Q, iK)

        self.assertTrue(np.allclose(C1, C2), "MAX DIFF: {:.4e}".format(np.abs(C1 - C2).max()))

    def test_cho_dot_ein(self):
        # attempt to compute the transformed covariance using cholesky decomposition

        # integrand
        # input moments
        mean_in = np.array([6500.4, 349.14, 1.8093, 6.7967, 0.6932])
        cov_in = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])

        f = ReentryVehicle2DTransition(GaussRV(5, mean_in, cov_in), GaussRV(3)).dyn_eval
        dim_in, dim_out = ReentryVehicle2DTransition.dim_state, 1

        # transform
        ker_par_mo = np.hstack((np.ones((dim_out, 1)), 25 * np.ones((dim_out, dim_in))))
        rbf_kernel = {'name': 'rbf', 'params': ker_par_mo}
        tf_so = GaussianProcessTransform(dim_in, dim_out, rbf_kernel)

        # Monte-Carlo for ground truth
        # tf_ut = UnscentedTransform(dim_in)
        # tf_ut.apply(f, mean_in, cov_in, np.atleast_1d(1), None)
        tf_mc = MonteCarloTransform(dim_in, 1000)
        mean_mc, cov_mc, ccov_mc = tf_mc.apply(f, mean_in, cov_in, np.atleast_1d(1))
        C_MC = cov_mc + np.outer(mean_mc, mean_mc.T)

        # evaluate integrand
        x = mean_in[:, None] + la.cholesky(cov_in).dot(tf_so.model.points)
        Y = np.apply_along_axis(f, 0, x, 1.0, None)

        # covariance via np.dot
        iK, Q = tf_so.model.iK, tf_so.model.Q
        C1 = iK.dot(Q).dot(iK)
        C1 = Y.dot(C1).dot(Y.T)

        # covariance via np.einsum
        C2 = np.einsum('ab, bc, cd', iK, Q, iK)
        C2 = np.einsum('ab,bc,cd', Y, C2, Y.T)

        # covariance via np.dot and cholesky
        K = tf_so.model.kernel.eval(tf_so.model.kernel.par, tf_so.model.points)
        L_lower = la.cholesky(K)
        Lq = la.cholesky(Q)
        phi = la.solve(L_lower, Lq)
        psi = la.solve(L_lower, Y.T)
        bet = psi.T.dot(phi)
        C3_dot = bet.dot(bet.T)
        C3_ein = np.einsum('ij, jk', bet, bet.T)

        logging.debug("MAX DIFF: {:.4e}".format(np.abs(C1 - C2).max()))
        logging.debug("MAX DIFF: {:.4e}".format(np.abs(C3_dot - C3_ein).max()))
        self.assertTrue(np.allclose(C1, C2), "MAX DIFF: {:.4e}".format(np.abs(C1 - C2).max()))
        self.assertTrue(np.allclose(C3_dot, C3_ein), "MAX DIFF: {:.4e}".format(np.abs(C3_dot - C3_ein).max()))
        self.assertTrue(np.allclose(C1, C3_dot), "MAX DIFF: {:.4e}".format(np.abs(C1 - C3_dot).max()))
