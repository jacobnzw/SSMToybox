# Content

This branch of repo contains source code to reproduce the results published in the article

Prüher, J., and Straka, 0. [Gaussian Process Quadrature Moment Transform](https://doi.org/10.1109/TAC.2017.2774444). IEEE Transactions on Automatic Control, 2017.

> Computation of moments of transformed random variables is a problem appearing in many engineering applications. The current methods for moment transformation are mostly based on the classical quadrature rules, which cannot account for the approximation errors. Our aim is to design a method for moment transformation for Gaussian random variables, which accounts for the error in the numerically computed mean. We employ an instance of Bayesian quadrature, called Gaussian process quadrature (GPQ), which allows us to treat the integral itself as a random variable, where the integral variance informs about the incurred integration error. Experiments on the coordinate transformation and nonlinear filtering examples show that the proposed GPQ moment transform performs better than the classical transforms.

All the code needed to reproduce the results is contained in the `paper_code` directory.

## Reproducing the Results
I was developing on Windows 10, so before launching, set the `PYTHONPATH` temporarily using

`set PYTHONPATH=%PYTHONPATH%;[your_drive]:\path\to\SSMToybox\`

and switch to the directory

`cd paper_code`

Executing `python gpq_tracking.py` will run the experiment comparing the *Gaussian Process Quadrature Kalman Filter* (GPQKF) with the *Unscented Kalman Filter* (UKF) on the radar tracking example.

Executing `python polar2cartesian.py` will run the experiment comparing performance of the Gaussian Process Quarature moment transform on a transformation from polar to Cartesian coordinates.


## Requirements

The following libraries are required to run the script

- Python 3
- NumPy
- SciPy
- GPy
- Matplotlib
- Pandas