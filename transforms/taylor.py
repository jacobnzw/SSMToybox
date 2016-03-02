from transforms.transform import MomentTransform


class Taylor1stOrder(MomentTransform):
    def apply(self, f, mean, cov, *args):
        mean_f = f(mean)
        jacobian_f = f(mean, dx=True)
        cov_fx = jacobian_f.dot(cov)
        cov_f = cov_fx.dot(jacobian_f.T)
        return mean_f, cov_f, cov_fx


class Taylor2ndOrder(MomentTransform):
    def apply(self, f, mean, cov, *args):
        pass


class Linear(MomentTransform):
    # would have to be implemented via first order Taylor, because for linear f(x) = Ax and h(x) = Hx,
    # the Jacobians would be A and H, which mean TaylorFirstOrder is exact inference for linear functions and,
    # in a sense, Kalman filter does not have to be explicitly implemented, because the ExtendedKalman becomes
    # Kalman for linear f() and h().
    def apply(self, f, mean, cov, *args):
        pass
