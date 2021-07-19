# SSM Toybox
Python 3 implementation of the nonlinear sigma-point Kalman filters based on Bayesian quadrature. The filter is understood to be composed of moment transforms, which lend it it's uniqueness, which is why the "moment transform" is the key term used troughout the publications and the toybox.

## Structure
The toybox itself is confined in the `ssmtoybox` folder together with the unit tests.

Under `research` is a code for reproducing results in the following publications (chronological order):
* `gpq`: Gaussian Process Quadrature Moment Transform [1]
* `gpqd`: Gaussian Process Quadrautre Moment Transform with Derivatives [2]
* `tpq`: Student's t-Process Quadrature Moment Transform [3]
* `bsq`: Bayes-Sard Quadrature Moment Transform [4]


### Build documentation
```
cd docs
sphinx-apidoc -o ./ ../ssmtoybox ../ssmtoybox/tests
make html
```


### Why toybox?
The aim of this project was mainly to provide a code base for testing ideas during my Ph.D. research related to the application of Bayesian quadrature for improvement of Kalman filter estimates in terms of crediblity. The code was never meant to be used seriously as a toolbox.


### References
[1]: [[DOI](http://dx.doi.org/10.1109/TAC.2017.2774444) | [PDF](https://arxiv.org/abs/1701.01356)] 
Prüher, J. & Straka, O. *Gaussian Process Quadrature Moment Transform*, IEEE Transactions on Automatic Control, 2017

[2]: [[DOI](https://doi.org/10.1109/MLSP.2016.7738903) | [PDF](https://ieeexplore.ieee.org/document/7738903)] Prüher, J., & Sarkka, S. (2016). *On the Use of Gradient Information in Gaussian Process Quadratures*. In 2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP) (pp. 1–6). 

[3]: [[DOI](http://dx.doi.org/10.23919/ICIF.2017.8009742) | [PDF](https://arxiv.org/abs/1703.05189)] 
Prüher, J.; Tronarp, F.; Karvonen, T.; Särkkä, S. & Straka, O. *Student-t Process Quadratures for Filtering of 
Non-linear Systems with Heavy-tailed Noise*, 20th International Conference on Information Fusion (Fusion), 1-8, 2017

[4]: [[DOI](https://doi.org/10.1109/TAC.2020.2991698) | [PDF](https://export.arxiv.org/pdf/1811.11474)] Pruher, J., Karvonen, T., Oates, C. J., Straka, O., & Sarkka, S. (2021). *Improved Calibration of Numerical Integration Error in Sigma-Point Filters*. IEEE Transactions on Automatic Control, 66(3), 1286–1292.

