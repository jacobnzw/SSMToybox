from transform import MomentTransform


class GaussianProcess(MomentTransform):
    # GPQ can work with any sigmas so it's probably better to pass in the unit sigmas
    # as an argument instead of creating them in init
    # BQ does not prescribe any sigma-point schemes (apart from minimium variance point sets)
    def __init__(self, unit_sp):
        self.unit_sp = unit_sp
        # set kernel hyper-parameters (manually or some principled method)
        self.hypers = 0  # log-hypers or hypers
        # weights (the most costly part)
        self.weights = self.weights()

    def apply(self, f, mean, cov, *args):
        pass

    @staticmethod
    def weights(unit_sp, hypers):
        pass

    def _min_int_var_sigmas(self):
        # minimum variance point set
        pass

    def _min_int_var_hypers(self):
        # finds hypers that minimize integral variance (these minimize MMD)
        pass

    def _min_logmarglik_hypers(self):
        # finds hypers by maximizing the marginal likelihood (empirical bayes)
        # the multiple output dimensions should be reflected in the log marglik
        pass

    def _min_intvar_logmarglik_hypers(self):
        # finds hypers by minimizing the sum of log-marginal likelihood and the integral variance objectives
        pass
