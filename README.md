# Content

This branch of repo contains source code to reproduce the results published in an article

>J. Prüher and S. Särkkä, "On the use of gradient information in Gaussian process quadratures," *2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP)*, Vietri sul Mare, 2016, pp. 1-6. [doi: 10.1109/MLSP.2016.7738903](https://doi.org/10.1109/MLSP.2016.7738903)

All the code needed to reproduce the results is contained in the `./paper_code` directory.

## Requirements

The following libraries are required to run the script

- NumPy
- SciPy
- GPy
- Matplotlib

## Reproducing the results
Switching to `./paper_code` directory and running the main script

`python mlsp2016_demo.py`

will launch all the experiments and save the resulting tables and figures in the same directory.
