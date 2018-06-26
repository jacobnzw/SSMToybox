# SSM Toybox
Python 3 implementation of the nonlinear sigma-point filters based on Bayesian quadrature, such as

* Gaussian Process Quadrature Kalman Filter [1]
* Student's t-Process Quadrature Kalman Filter [2]

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


### References
[1]: [[DOI](http://dx.doi.org/10.1109/TAC.2017.2774444) | [PDF](https://arxiv.org/abs/1701.01356)] 
Prüher, J. and Straka, O. Gaussian Process Quadrature Moment Transform, IEEE Transactions on Automatic Control, 2017

[2]: [[DOI](http://dx.doi.org/10.23919/ICIF.2017.8009742) | [PDF](https://arxiv.org/abs/1703.05189)] 
Prüher, J.; Tronarp, F.; Karvonen, T.; Särkkä, S. and Straka, O. Student-t Process Quadratures for Filtering of 
Non-linear Systems with Heavy-tailed Noise, 20th International Conference on Information Fusion (Fusion), 1-8, 2017 

