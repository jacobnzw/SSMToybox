# SSM Toybox
Python 3 implementation of the nonlinear sigma-point filters based on Bayesian quadrature, such as

* Gaussian Process Quadrature Kalman Filter
* Student's t-Process Quadrature Kalman Filter

Included are also the well-known classical nonlinear Kalman filters such as:

* Extended Kalman Filter
* Unscented Kalman Filter
* Cubature Kalman Filter
* Gauss-Hermite Kalman Filter


### Build documentation

```
cd docs
sphinx-apidoc -o ./ ../ssmtoybox ../ssmtoybox/tests
make html
```


### Why toybox?

Because 'toolbox' sounds too serious :-).