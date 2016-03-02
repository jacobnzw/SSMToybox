from __future__ import division


class MomentTransform(object):

    def apply(self, f, mean, cov, *args):
        raise NotImplementedError

# TODO: implement GPQ+, GPQ+D, GPQ+TD, TPQ+ transforms (adopt from BQ repo)
# Statistically linearized is pain to use (needs expectations of nonlinearities)
