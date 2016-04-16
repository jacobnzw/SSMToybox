from transforms.transform import MomentTransform
import numpy as np
import numpy.linalg as la


class Taylor1stOrder(MomentTransform):
    def __init__(self, dim):
        self.dim = dim

    def apply(self, f, mean, cov, pars):
        mean_f = f(mean, pars)
        jacobian_f = f(mean, pars, dx=True)
        jacobian_f = jacobian_f.reshape(len(mean_f), self.dim)
        cov_fx = jacobian_f.dot(cov)
        cov_f = cov_fx.dot(jacobian_f.T)
        return mean_f, cov_f, cov_fx


class Taylor2ndOrder(MomentTransform):
    def apply(self, f, mean, cov, pars):
        pass


class TaylorGPQD(MomentTransform):
    """Transformation equivalent to GPQ+D w/ RBF kernel, single sigma-point at zero and substitution x = m + z in the
    integral. For el --> infinity the transform converges to Taylor1stOrder transform.
    """

    def __init__(self, dim, alpha=1.0, el=1.0):
        self.dim = dim
        self.alpha = alpha
        self.ell = el
        self.Lam = np.diag(el ** 2 * np.ones(dim))
        self.iLam = np.diag(el ** -2 * np.ones(dim))
        self.eye_d = np.eye(dim)
        # lists for logging average model variance and posterior integral variance
        self.mvar_list = []
        self.ivar_list = []

    def apply(self, f, mean, cov, pars):
        # TODO: equations can be optimized further
        # wm = la.det(cov.dot(self.iLam) + self.eye_d) ** -0.5
        wm = la.det(self.iLam.dot(cov) + self.eye_d) ** -0.5
        fm = f(mean, pars)
        mean_f = wm * fm
        jacobian_f = f(mean, pars, dx=True)
        jacobian_f = jacobian_f.reshape(len(mean_f), self.dim)
        # wc = la.det(cov.dot(2 * self.iLam) + self.eye_d) ** -0.5
        wc = la.det(2 * self.iLam.dot(cov) + self.eye_d) ** -0.5
        Wc = 0.5 * self.Lam.dot(la.inv(0.5 * self.Lam + cov)).dot(cov)
        model_var = self.alpha ** 2 - self.alpha ** 2 * wc * (1 + np.trace(Wc.dot(self.iLam)))
        integ_var = self.alpha ** 2 * wc - wm ** 2
        self.mvar_list.append(model_var)
        self.ivar_list.append(integ_var)
        cov_f = wc * (np.outer(fm, fm) + jacobian_f.dot(Wc).dot(jacobian_f.T)) - np.outer(mean_f, mean_f) + model_var
        cov_fx = self.Lam.dot(la.inv(self.Lam + cov)).dot(cov).dot(jacobian_f.T)
        return mean_f, cov_f, cov_fx


class Linear(MomentTransform):
    # would have to be implemented via first order Taylor, because for linear f(x) = Ax and h(x) = Hx,
    # the Jacobians would be A and H, which mean TaylorFirstOrder is exact inference for linear functions and,
    # in a sense, Kalman filter does not have to be explicitly implemented, because the ExtendedKalman becomes
    # Kalman for linear f() and h().
    def apply(self, f, mean, cov, pars):
        pass
