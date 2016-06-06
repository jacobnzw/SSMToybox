from unittest import TestCase
from kernel import RBF, Affine
from transforms.model import *
import numpy as np
import numpy.linalg as la
from numpy import newaxis as na

# fcn = lambda x: np.sin((x + 1) ** -1)
fcn = lambda x: x ** 2


# fcn = lambda x: x


class GPModelTest(TestCase):
    def test_init(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        phyp = {'alpha': 1.0}
        GaussianProcess(1)
        GaussianProcess(1, kernel='rbf', points='ut', kern_hyp=khyp, point_hyp=phyp)

    def test_plotting(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        model = GaussianProcess(1, kern_hyp=khyp)
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        model.plot_model(xtest, y, f)


class TPModelTest(TestCase):
    def test_init(self):
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(1)}
        phyp = {'alpha': 1.0}
        StudentTProcess(1)
        StudentTProcess(1, kernel='rbf', points='ut', kern_hyp=khyp, point_hyp=phyp)

    def test_plotting(self):
        dim = 1
        khyp = {'alpha': 1.0, 'el': 3.0 * np.ones(dim, )}
        model = StudentTProcess(dim, kern_hyp=khyp)
        xtest = np.linspace(-5, 5, 50)[na, :]
        y = fcn(model.points)
        f = fcn(xtest)
        model.plot_model(xtest, y, f)
