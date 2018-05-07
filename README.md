# Content

This branch of repo contains source code to reproduce the results published in the article

J. Prüher, F. Tronarp, T. Karvonen, S. Särkkä and O. Straka, [Student-t process quadratures for filtering of non-linear systems with heavy-tailed noise](https://dx.doi.org/10.23919/ICIF.2017.8009742), 20th International Conference on Information Fusion (Fusion), Xi'an, 2017, pp. 1-8.

**Abstract**
> The aim of this article is to design a moment transformation for Student-t distributed random variables, which is able to account for the error in the numerically computed mean. We employ Student-t process quadrature, an instance of Bayesian quadrature, which allows us to treat the integral itself as a random variable whose variance provides information about the incurred integration error. Advantage of the Student-t process quadrature over the traditional Gaussian process quadrature, is that the integral variance depends also on the function values, allowing for a more robust modelling of the integration error. The moment transform is applied in nonlinear sigma-point filtering and evaluated on two numerical examples, where it is shown to outperform the state-of-the-art moment transforms.

All the code needed to reproduce the results is contained in the `fusion_paper` directory.

## Reproducing the Results
I was developing on Windows 10, so before launching, set the `PYTHONPATH` temporarily using

`set PYTHONPATH=%PYTHONPATH%;[your_drive]:\path\to\SSMToybox\`

and switch to the directory

`cd fusion_paper`

Executing `python synthetic.py` will run the experiment comparing the *Student's t-Process Quadrature Student's Filter* (TPQSF) with the *Student's Filter* (SF) in the radar tracking with glint noise scenario.

## Requirements

The following libraries are required to run the script

- Python 3
- NumPy
- SciPy
- GPy
- Matplotlib
- Pandas