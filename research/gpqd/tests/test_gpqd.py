import numpy as np

from scipy.linalg import cho_factor, cho_solve, solve
from numpy.linalg import det
from numpy import newaxis as na

from unittest import TestCase

from research.gpqd.gpqd_base import GaussianProcessDerModel, RBFGaussDer
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
        if Ns == N:
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
        cho_B = cho_factor(B)
        t = cho_solve(cho_B, inn)  # dot(inn, inv(B)) # (x-m)^T*iLam*(P+Lambda)^-1  # (D, N)
        l = np.exp(-0.5 * np.sum(inn * t, 0))  # (N, 1)
        q = (alpha ** 2 / np.sqrt(det(B))) * l  # (N, 1)
        Sig_q = cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        mu_q = iiLam.dot(eta)  # (D,N)
        r = q[na, which_der] * iiLam.dot(mu_q[:, which_der] - x[:, which_der])  # -t.dot(iLam) * q  # (D, N)

        return r.T.ravel()  # (1, num_der*dim)

    @staticmethod
    def expected_exp_x_xdkx(x, alpha=1.0, el=1.0, which_der=None):
        dim, num_pts = x.shape
        which_der = np.arange(num_pts) if which_der is None else which_der
        num_der = len(which_der)
        el = np.asarray(dim * [el])
        eye_d = np.eye(dim)
        iLam = np.diag(el ** -1)  # sqrt(Lambda^-1)
        iiLam = np.diag(el ** -2)  # Lambda^-1

        inn = iLam.dot(x)  # (x-m)^T*iLam  # (N, D)
        B = iiLam + eye_d
        cho_B = cho_factor(B)
        t = cho_solve(cho_B, inn)
        l = np.exp(-0.5 * np.sum(inn * t, 0))  # (N, 1)
        q = (alpha ** 2 / np.sqrt(np.linalg.det(B))) * l  # (N, 1)
        Sig_q = cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        mu_q = iiLam.dot(eta)  # (D,N)
        r = q[na, which_der] * iiLam.dot(mu_q[:, which_der] - x[:, which_der])  # -t.dot(iLam) * q  # (D, N)

        #  quantities for cross-covariance "weights"
        iLamSig = iiLam.dot(Sig_q)  # (D,D)
        r_tilde = np.empty((dim, num_der * dim))
        for i in range(num_der):
            i_d = which_der[i]
            r_tilde[:, i * dim:i * dim + dim] = q[i_d] * iLamSig + np.outer(mu_q[:, i_d], r[:, i].T)

        return r_tilde  # (dim, num_der*dim)

    @staticmethod
    def expected_exp_x_kxdkx(x, alpha=1.0, el=1.0, which_der=None):
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
        cho_B = cho_factor(B)
        Sig_q = cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        # quantities for covariance weights
        zet = 2 * np.log(alpha) - 0.5 * np.sum(inn * inn, 0)  # (D,N) 2log(alpha) - 0.5*(x-m)^T*Lambda^-1*(x-m)
        inn = iiLam.dot(x)  # inp / el[:, na]**2
        R = 2 * iiLam + eye_d  # 2P*Lambda^-1 + I
        Q = (1.0 / np.sqrt(det(R))) * np.exp((zet[:, na] + zet[:, na].T) +
                                             maha(inn.T, -inn.T, V=0.5 * solve(R, eye_d)))  # (N,N)
        cho_LamSig = cho_factor(Lam + Sig_q)
        eta_tilde = iiLam.dot(cho_solve(cho_LamSig, eta))  # Lambda^-1(Lambda+Sig_q)^-1*eta
        mu_Q = eta_tilde[..., na] + eta_tilde[:, na, :]  # (D,N_der,N) pairwise sum of pre-multiplied eta's

        E_dfff = np.empty((num_der * dim, num_pts))
        for i in range(num_der):
            for j in range(num_pts):
                istart, iend = i * dim, i * dim + dim
                i_d = which_der[i]
                E_dfff[istart:iend, j] = Q[i_d, j] * (mu_Q[:, i_d, j] - inn[:, i_d])

        return E_dfff.T  # (num_der*dim, num_pts)

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
        cho_B = cho_factor(B)
        Sig_q = cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(x)  # (D,N) Sig_q*x
        # quantities for covariance weights
        zet = 2 * np.log(alpha) - 0.5 * np.sum(inn * inn, 0)  # (D,N) 2log(alpha) - 0.5*(x-m)^T*Lambda^-1*(x-m)
        inn = iiLam.dot(x)  # inp / el[:, na]**2
        R = 2 * iiLam + eye_d  # 2P*Lambda^-1 + I
        Q = (1.0 / np.sqrt(det(R))) * np.exp((zet[:, na] + zet[:, na].T) +
                                             maha(inn.T, -inn.T, V=0.5 * solve(R, eye_d)))  # (N,N)
        cho_LamSig = cho_factor(Lam + Sig_q)
        Sig_Q = cho_solve(cho_LamSig, Sig_q).dot(iiLam)  # (D,D) Lambda^-1 (Lambda*(Lambda+Sig_q)^-1*Sig_q) Lambda^-1
        eta_tilde = iiLam.dot(cho_solve(cho_LamSig, eta))  # Lambda^-1(Lambda+Sig_q)^-1*eta
        mu_Q = eta_tilde[..., na] + eta_tilde[:, na, :]  # (D,N_der,N) pairwise sum of pre-multiplied eta's

        E_dffd = np.empty((num_der * dim, num_der * dim))
        for i in range(num_der):
            for j in range(num_der):
                istart, iend = i * dim, i * dim + dim
                jstart, jend = j * dim, j * dim + dim
                i_d, j_d = which_der[i], which_der[j]
                T = np.outer((inn[:, i_d] - mu_Q[:, i_d, j_d]), (inn[:, j_d] - mu_Q[:, i_d, j_d]).T) + Sig_Q
                E_dffd[istart:iend, jstart:jend] = Q[i_d, j_d] * T

        return E_dffd  # (num_der*dim, num_der*dim)

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

    def test_exp_x_xdkx(self):
        kernel, kernel_par, x = self.kernel, self.kernel_par, self.x

        out = kernel.exp_x_xdkx(kernel_par, x)
        exp = RBFGaussDerKernelTest.expected_exp_x_xdkx(x, alpha=kernel_par[0, 0], el=kernel_par[0, 1])

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


class GaussianProcessDerModelTest(TestCase):

    @staticmethod
    def weights_rbf_der(unit_sp, alpha=1.0, el=1.0, which_der=None):
        d, n = unit_sp.shape
        which_der = which_der if which_der is not None else np.arange(n)

        el = np.asarray(d * [el])
        assert len(el) == d
        i_der = which_der  # shorthand for indexes of points with derivatives
        n_der = len(i_der)  # # points w/ derivatives
        assert n_der <= n  # # points w/ derivatives must be <= # points
        # pre-allocation for convenience
        eye_d, eye_n, eye_y = np.eye(d), np.eye(n), np.eye(n + d * n_der)

        K = RBFGaussDerKernelTest.expected_eval(unit_sp, unit_sp, alpha=alpha, el=el, which_der=i_der)
        iK = cho_solve(cho_factor(K + 1e-8 * eye_y), eye_y)  # invert kernel matrix BOTTLENECK
        Lam = np.diag(el ** 2)
        iLam = np.diag(el ** -1)  # sqrt(Lambda^-1)
        iiLam = np.diag(el ** -2)  # Lambda^-1
        inn = iLam.dot(unit_sp)  # (x-m)^T*iLam  # (N, D)
        B = iiLam + eye_d  # P*Lambda^-1+I, (P+Lam)^-1 = Lam^-1*(P*Lam^-1+I)^-1 # (D, D)
        cho_B = cho_factor(B)
        t = cho_solve(cho_B, inn)  # dot(inn, inv(B)) # (x-m)^T*iLam*(P+Lambda)^-1  # (D, N)
        l = np.exp(-0.5 * np.sum(inn * t, 0))  # (N, 1)
        q = (alpha ** 2 / np.sqrt(np.linalg.det(B))) * l  # (N, 1)
        Sig_q = cho_solve(cho_B, eye_d)  # B^-1*I
        eta = Sig_q.dot(unit_sp)  # (D,N) Sig_q*x
        mu_q = iiLam.dot(eta)  # (D,N)
        r = q[na, i_der] * iiLam.dot(mu_q[:, i_der] - unit_sp[:, i_der])  # -t.dot(iLam) * q  # (D, N)
        q_tilde = np.hstack((q.T, r.T.ravel()))  # (1, N + n_der*D)

        # weights for mean
        wm = q_tilde.dot(iK)

        #  quantities for cross-covariance "weights"
        iLamSig = iiLam.dot(Sig_q)  # (D,D)
        r_tilde = np.empty((d, n_der * d))
        for i in range(n_der):
            i_d = i_der[i]
            r_tilde[:, i * d:i * d + d] = q[i_d] * iLamSig + np.outer(mu_q[:, i_d], r[:, i].T)
        R_tilde = np.hstack((q[na, :] * mu_q, r_tilde))  # (D, N+N*D)

        # input-output covariance (cross-covariance) "weights"
        Wcc = R_tilde.dot(iK)  # (D, N+N*D)

        # quantities for covariance weights
        zet = 2 * np.log(alpha) - 0.5 * np.sum(inn * inn, 0)  # (D,N) 2log(alpha) - 0.5*(x-m)^T*Lambda^-1*(x-m)
        inn = iiLam.dot(unit_sp)  # inp / el[:, na]**2
        R = 2 * iiLam + eye_d  # 2P*Lambda^-1 + I
        Q = (1.0 / np.sqrt(det(R))) * np.exp((zet[:, na] + zet[:, na].T) +
                                             maha(inn.T, -inn.T, V=0.5 * solve(R, eye_d)))  # (N,N)
        cho_LamSig = cho_factor(Lam + Sig_q)
        Sig_Q = cho_solve(cho_LamSig, Sig_q).dot(iiLam)  # (D,D) Lambda^-1 (Lambda*(Lambda+Sig_q)^-1*Sig_q) Lambda^-1
        eta_tilde = iiLam.dot(cho_solve(cho_LamSig, eta))  # Lambda^-1(Lambda+Sig_q)^-1*eta
        mu_Q = eta_tilde[..., na] + eta_tilde[:, na, :]  # (D,N_der,N) pairwise sum of pre-multiplied eta's

        E_dfff = np.empty((n_der * d, n))
        for i in range(n_der):
            for j in range(n):
                istart, iend = i * d, i * d + d
                i_d = i_der[i]
                E_dfff[istart:iend, j] = Q[i_d, j] * (mu_Q[:, i_d, j] - inn[:, i_d])

        E_dffd = np.empty((n_der * d, n_der * d))
        for i in range(n_der):
            for j in range(n_der):
                istart, iend = i * d, i * d + d
                jstart, jend = j * d, j * d + d
                i_d, j_d = i_der[i], i_der[j]
                T = np.outer((inn[:, i_d] - mu_Q[:, i_d, j_d]), (inn[:, j_d] - mu_Q[:, i_d, j_d]).T) + Sig_Q
                E_dffd[istart:iend, jstart:jend] = Q[i_d, j_d] * T
        Q_tilde = np.vstack((np.hstack((Q, E_dfff.T)), np.hstack((E_dfff, E_dffd))))  # (N + N_der*D, N + N_der*D)

        # weights for covariance
        iKQ = iK.dot(Q_tilde)
        Wc = iKQ.dot(iK)

        return wm, Wc, Wcc

    def test_weights(self):
        dim = 2
        kernel_par = np.array([[1.0] + dim*[2.0]])
        model = GaussianProcessDerModel(dim, kernel_par, 'ut')

        out_wm, out_wc, out_wcc, _, _ = model.bq_weights(kernel_par)
        exp_wm, exp_wc, exp_wcc = GaussianProcessDerModelTest.weights_rbf_der(model.points,
                                                                              alpha=kernel_par[0, 0],
                                                                              el=kernel_par[0, 1])
        # check shapes
        self.assertEqual(out_wm.shape, exp_wm.shape)
        self.assertEqual(out_wc.shape, exp_wc.shape)
        self.assertEqual(out_wcc.shape, exp_wcc.shape)

        # check values
        self.assertTrue(np.allclose(out_wm, out_wm))
        self.assertTrue(np.allclose(out_wc, out_wc))
        self.assertTrue(np.allclose(out_wcc, out_wcc))
