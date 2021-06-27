import numpy as np
import scipy as sp

from numpy import newaxis as na
from unittest import TestCase
from research.gpqd.mlsp2016_demo import RBFGaussDer
from ssmtoybox.utils import maha
from ssmtoybox.mtran import UnscentedTransform


class RBFGaussDerKernelTest(TestCase):

    @staticmethod
    def expected_eval(xs, x, alpha=10.0, el=0.7, which_der=None):
        """RBF kernel w/ derivatives."""
        x, xs = np.atleast_2d(x), np.atleast_2d(xs)
        D, N = x.shape
        Ds, Ns = xs.shape
        assert Ds == D
        which_der = np.arange(N) if which_der is None else which_der
        Nd = len(which_der)  # points w/ derivative observations
        # extract hypers
        # alpha, el, jitter = hypers['sig_var'], hypers['lengthscale'], hypers['noise_var']
        iLam = np.diag(el ** -1 * np.ones(D))
        iiLam = np.diag(el ** -2 * np.ones(D))

        x = iLam.dot(x)  # sqrt(Lambda^-1) * X
        xs = iLam.dot(xs)
        Kff = np.exp(2 * np.log(alpha) - 0.5 * maha(xs.T, x.T))  # cov(f(xi), f(xj))
        x = iLam.dot(x)  # Lambda^-1 * X
        xs = iLam.dot(xs)
        XmX = xs[..., na] - x[:, na, :]  # pair-wise differences
        Kfd = np.zeros((Ns, D * Nd))  # cov(f(xi), df(xj))
        Kdd = np.zeros((D * Nd, D * Nd))  # cov(df(xi), df(xj))
        for i in range(Ns):
            for j in range(Nd):
                jstart, jend = j * D, j * D + D
                j_d = which_der[j]
                Kfd[i, jstart:jend] = Kff[i, j_d] * XmX[:, i, j_d]
        for i in range(Nd):
            for j in range(Nd):
                istart, iend = i * D, i * D + D
                jstart, jend = j * D, j * D + D
                i_d, j_d = which_der[i], which_der[j]  # indices of points with derivatives
                Kdd[istart:iend, jstart:jend] = Kff[i_d, j_d] * (iiLam - np.outer(XmX[:, i_d, j_d], XmX[:, i_d, j_d]))
        if Ns == N:  # TODO: xs should be kwarg just like in GPy to properly recognize if test points are used
            return np.vstack((np.hstack((Kff, Kfd)), np.hstack((Kfd.T, Kdd))))
        else:
            return np.hstack((Kff, Kfd))

    @staticmethod
    def expected_exp_x_dkx(x, alpha=1.0, el=1.0, which_der=None):
        dim, num_pts = x.shape
        which_der = which_der if which_der is not None else np.arange(num_pts)
        eye_d = np.eye(dim)
        el = np.asarray(dim * [el])
        iLam = np.diag(el ** -1)  # sqrt(Lambda^-1)
        iiLam = np.diag(el ** -2)  # Lambda^-1
        inn = iLam.dot(x)  # (x-m)^T*iLam  # (N, D)
        B = iiLam + eye_d  # P*Lambda^-1+I, (P+Lam)^-1 = Lam^-1*(P*Lam^-1+I)^-1 # (D, D)
        cho_B = sp.linalg.cho_factor(B)
        t = sp.linalg.cho_solve(cho_B, inn)  # dot(inn, inv(B)) # (x-m)^T*iLam*(P+Lambda)^-1  # (D, N)
        l = np.exp(-0.5 * np.sum(inn * t, 0))  # (N, 1)
        q = (alpha ** 2 / np.sqrt(np.linalg.det(B))) * l  # (N, 1)
        Sig_q = sp.linalg.cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        mu_q = iiLam.dot(eta)  # (D,N)
        r = q[na, which_der] * iiLam.dot(mu_q[:, which_der] - x[:, which_der])  # -t.dot(iLam) * q  # (D, N)

        return np.hstack((q.T, r.T.ravel()))  # q_tilde (1, N + n_der*D)

    @staticmethod
    def expected_exp_x_kxdkx(x, alpha=1.0, el=1.0, which_der=None):
        dim, num_pts = x.shape
        which_der = which_der if which_der is not None else np.arange(num_pts)
        num_der = len(which_der)
        eye_d = np.eye(dim)
        el = np.asarray(dim * [el])
        iLam = np.diag(el ** -1)  # sqrt(Lambda^-1)
        iiLam = np.diag(el ** -2)  # Lambda^-1

        inn = iLam.dot(x)  # (x-m)^T*iLam  # (N, D)
        B = iiLam + eye_d  # P*Lambda^-1+I, (P+Lam)^-1 = Lam^-1*(P*Lam^-1+I)^-1 # (D, D)
        cho_B = sp.linalg.cho_factor(B)
        t = sp.linalg.cho_solve(cho_B, inn)  # dot(inn, inv(B)) # (x-m)^T*iLam*(P+Lambda)^-1  # (D, N)
        l = np.exp(-0.5 * np.sum(inn * t, 0))  # (N, 1)
        q = (alpha ** 2 / np.sqrt(np.linalg.det(B))) * l  # (N, 1)
        Sig_q = sp.linalg.cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        mu_q = iiLam.dot(eta)  # (D,N)
        r = q[na, which_der] * iiLam.dot(mu_q[:, which_der] - x[:, which_der])  # -t.dot(iLam) * q  # (D, N)

        #  quantities for cross-covariance "weights"
        iLamSig = iiLam.dot(Sig_q)  # (D,D)
        r_tilde = np.empty((dim, num_der * dim))
        for i in range(num_der):
            i_d = which_der[i]
            r_tilde[:, i * dim:i * dim + dim] = q[i_d] * iLamSig + np.outer(mu_q[:, i_d], r[:, i].T)

        return np.hstack((q[na, :] * mu_q, r_tilde))  # R_tilde (D, N+N*D)

    @staticmethod
    def expected_exp_x_dkxdkx(x, alpha=1.0, el=1.0, which_der=None):
        dim, num_pts = x.shape
        which_der = which_der if which_der is not None else np.arange(num_pts)
        num_der = len(which_der)
        eye_d = np.eye(dim)
        el = np.asarray(dim * [el])
        Lam = np.diag(el ** 2)
        iLam = np.diag(el ** -1)  # sqrt(Lambda^-1)
        iiLam = np.diag(el ** -2)  # Lambda^-1

        inn = iLam.dot(x)  # (x-m)^T*iLam  # (N, D)
        B = iiLam + eye_d  # P*Lambda^-1+I, (P+Lam)^-1 = Lam^-1*(P*Lam^-1+I)^-1 # (D, D)
        cho_B = sp.linalg.cho_factor(B)
        Sig_q = sp.linalg.cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        # quantities for covariance weights
        zet = 2 * np.log(alpha) - 0.5 * np.sum(inn * inn, 0)  # (D,N) 2log(alpha) - 0.5*(x-m)^T*Lambda^-1*(x-m)
        inn = iiLam.dot(x)  # inp / el[:, na]**2
        R = 2 * iiLam + eye_d  # 2P*Lambda^-1 + I
        Q = (1.0 / np.sqrt(np.linalg.det(R))) * np.exp((zet[:, na] + zet[:, na].T) +
                                             maha(inn.T, -inn.T, V=0.5 * sp.linalg.solve(R, eye_d)))  # (N,N)
        cho_LamSig = sp.linalg.cho_factor(Lam + Sig_q)
        Sig_Q = sp.linalg.cho_solve(cho_LamSig, Sig_q).dot(iiLam)  # (D,D) Lambda^-1 (Lambda*(Lambda+Sig_q)^-1*Sig_q) Lambda^-1
        eta_tilde = iiLam.dot(sp.linalg.cho_solve(cho_LamSig, eta))  # Lambda^-1(Lambda+Sig_q)^-1*eta
        mu_Q = eta_tilde[..., na] + eta_tilde[:, na, :]  # (D,N_der,N) pairwise sum of pre-multiplied eta's

        E_dfff = np.empty((num_der * dim, num_pts))
        for i in range(num_der):
            for j in range(num_pts):
                istart, iend = i * dim, i * dim + dim
                i_d = which_der[i]
                E_dfff[istart:iend, j] = Q[i_d, j] * (mu_Q[:, i_d, j] - inn[:, i_d])

        E_dffd = np.empty((num_der * dim, num_der * dim))
        for i in range(num_der):
            for j in range(num_der):
                istart, iend = i * dim, i * dim + dim
                jstart, jend = j * dim, j * dim + dim
                i_d, j_d = which_der[i], which_der[j]
                T = np.outer((inn[:, i_d] - mu_Q[:, i_d, j_d]), (inn[:, j_d] - mu_Q[:, i_d, j_d]).T) + Sig_Q
                E_dffd[istart:iend, jstart:jend] = Q[i_d, j_d] * T

        return np.vstack((np.hstack((Q, E_dfff.T)), np.hstack((E_dfff, E_dffd))))  # Q_tilde (N + N_der*D, N + N_der*D)

    def setUp(self) -> None:
        self.dim = 2
        self.x = UnscentedTransform.unit_sigma_points(self.dim)
        self.kernel_par = np.array([[1.0] + self.dim * [2.0]])
        self.kernel = RBFGaussDer(self.dim, self.kernel_par)

    def test_eval(self):
        kernel, kernel_par, x = self.kernel, self.kernel_par, self.x

        out = kernel.eval(kernel_par, x)
        exp = RBFGaussDerKernelTest.expected_eval(x, x, alpha=1.0, el=2.0)

        self.assertEqual(out.shape, exp.shape)
        self.assertTrue(np.allclose(out, exp))

    def test_exp_x_dkx(self):
        kernel, kernel_par, x = self.kernel, self.kernel_par, self.x

        out = kernel.exp_x_dkx(kernel_par, x)
        exp = RBFGaussDerKernelTest.expected_exp_x_dkx(x, alpha=kernel_par[0, 0], el=kernel_par[0, 1])

        self.assertEqual(out.shape, exp.shape)
        self.assertTrue(np.allclose(out, exp))

    def test_exp_x_kxdkx(self):
        kernel, kernel_par, x = self.kernel, self.kernel_par, self.x

        out = kernel.exp_x_kxdkx(kernel_par, x)
        exp = RBFGaussDerKernelTest.expected_exp_x_kxdkx(x, alpha=kernel_par[0, 0], el=kernel_par[0, 1])

        self.assertEqual(out.shape, exp.shape)
        self.assertTrue(np.allclose(out, exp))

    def test_exp_x_dkxdkx(self):
        kernel, kernel_par, x = self.kernel, self.kernel_par, self.x

        out = kernel.exp_x_dkxdkx(kernel_par, x)
        exp = RBFGaussDerKernelTest.expected_exp_x_dkxdkx(x, alpha=kernel_par[0, 0], el=kernel_par[0, 1])

        self.assertEqual(out.shape, exp.shape)
        self.assertTrue(np.allclose(out, exp))