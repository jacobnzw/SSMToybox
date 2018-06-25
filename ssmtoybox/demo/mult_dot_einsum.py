import numpy as np
import numpy.linalg as la
from scipy.linalg import cho_factor, cho_solve
from ssmod import ReentryRadar
from bq.bqmtran import GPQ
from mtran import MonteCarlo
from unittest import TestCase


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
                        "MAX DIFF: {:.4e}".format(np.abs(b.dot(A) - B.dot(A)[0, :]).max()))

    def test_einsum_dot(self):
        # einsum and dot give different results?

        dim_in, dim_out = 2, 1
        ker_par_mo = np.hstack((np.ones((dim_out, 1)), 1 * np.ones((dim_out, dim_in))))
        tf_mo = GPQ(dim_in, ker_par_mo, points='sr')
        iK, Q = tf_mo.iK, tf_mo.Q

        C1 = iK.dot(Q).dot(iK)
        C2 = np.einsum('ab, bc, cd', iK, Q, iK)

        self.assertTrue(np.allclose(C1, C2), "MAX DIFF: {:.4e}".format(np.abs(C1 - C2).max()))

    def test_cho_dot_ein(self):
        # attempt to compute the transformed covariance using cholesky decomposition

        # integrand
        ssm = ReentryRadar()
        f = ssm.dyn_eval
        dim_in, dim_out = ssm.xD, 1

        # input moments
        mean_in, cov_in = ssm.pars['x0_mean'], ssm.pars['x0_cov']

        # transform
        ker_par_mo = np.hstack((np.ones((dim_out, 1)), 25 * np.ones((dim_out, dim_in))))
        tf_so = GPQ(dim_in, ker_par_mo, points='sr')

        # Monte-Carlo for ground truth
        tf_mc = MonteCarlo(dim_in, 1000)
        mean_mc, cov_mc, ccov_mc = tf_mc.apply(f, mean_in, cov_in, None)
        C_MC = cov_mc + np.outer(mean_mc, mean_mc.T)

        # evaluate integrand
        x = mean_in[:, None] + la.cholesky(cov_in).dot(tf_so.model.points)
        Y = np.apply_along_axis(f, 0, x, None)

        # covariance via np.dot
        iK, Q = tf_so.iK, tf_so.Q
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

        print("MAX DIFF: {:.4e}".format(np.abs(C1 - C2).max()))
        print("MAX DIFF: {:.4e}".format(np.abs(C3_dot - C3_ein).max()))
        self.assertTrue(np.allclose(C1, C2), "MAX DIFF: {:.4e}".format(np.abs(C1 - C2).max()))
        self.assertTrue(np.allclose(C3_dot, C3_ein), "MAX DIFF: {:.4e}".format(np.abs(C3_dot - C3_ein).max()))
        self.assertTrue(np.allclose(C1, C3_dot), "MAX DIFF: {:.4e}".format(np.abs(C1 - C3_dot).max()))
