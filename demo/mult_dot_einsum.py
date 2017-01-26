import numpy as np
import numpy.linalg as la
from numpy import newaxis as na
from scipy.linalg import cho_factor, cho_solve
from models.tracking import ReentryRadar
from transforms.bayesquad import GPQ, GPQMO
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
