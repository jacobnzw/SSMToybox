from transform import MomentTransform


class GaussianProcess(MomentTransform):
    def __init__(self):
        # call some unit sigma-point method
        self.unit_sp = 0
        # set kernel hyper-parameters (manually or some principled method)
        self.hypers = 0  # log-hypers or hypers
        # weights (the most costly part)
        self.weights = 0

    def apply(self, f, mean, cov, *args):
        pass

    def weights(self):
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
