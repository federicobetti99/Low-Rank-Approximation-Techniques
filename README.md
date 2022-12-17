# Low Rank Approximation Techniques
This repository contains the numerical experiments carried out for the leverage score sampling project, carried out in the context of the MATH-403 Low-Rank Approximation Techniques course at EPFL.

## Repository description
- `code` - Implementation of the experiments
  - `utils` - Subfolder with utilities functions
  - `main.m` - Main script to run all the experiments
  - `lambda_tuning.m` - Adaptive regularization parameter experiment
  - `comparison.m` - Comparison between sampling strategies
- `figures` - Plots of the obtained results by running the three scripts above
  
## Reproducibility of the results
We provide two Matlab scripts to reproduce the results shown in the report. The script `main.m` carries out a comparison between uniform sampling, columns norm sampling and ridge leverage scores sampling for a Hilbert matrix of size 100, and produces a plot of the average projection error for all the strategies. The script `lambda_tuning.m` considers instead a full column rank matrix and shows that for a wise choice of the regularization parameter ridge leverage scores are basically equivalent to sampling from the rows of the k-truncated matrix of the right singular vectors of A.
  
## Authors
- Federico Betti
- Fabio Zoccolan
