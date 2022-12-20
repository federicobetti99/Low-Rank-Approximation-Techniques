# Low Rank Approximation Techniques
This repository contains the numerical experiments carried out for the leverage score sampling project, carried out in the context of the MATH-403 Low-Rank Approximation Techniques course at EPFL.

## Repository description
- `code` - Implementation of the experiments
  - `utils` - Subfolder with utilities functions
  - `main.m` - Main script to run all the experiments
  - `comparison.m` - Comparison between sampling strategies
  - `dist_scores.m` - Plot of cumulative sum of sketching distributions
  - `lambda_tuning.m` - Adaptive regularization parameter experiment
- `figures` - Plots of the obtained results
- `report.pdf` - Report of the obtained results
  
## Reproducibility of the results
We provide a unique Matlab script `main.m` to reproduce the results shown in the report. The latter runs the three scripts `comparison.m`, `dist_scores.m` and `lambda_tuning.m`, thus producing the three plots presented in the report in the same order as they appear in the latter. In particular:

1. `comparison.m` compares uniform column sampling, columns norm sampling and ridge leverage scores sampling for a Hilbert matrix of size 100, and produces a plot of the average projection error for the considered sketching distributions.
2. `dist_scores.m` produces a plot of the cumulative sum of the sketching distribution tested in `comparison.m` to give further insight on their different performance in the latter experiment.
3. `lambda_tuning.m` shows the similarity between the performance of ridge leverage scores sampling (for a wise choice of the regularization parameter) and sampling proportionally to the norm of the rows of the k-truncated right factor in the SVD of A for a full column rank matrix with slow singular value decay.
  
## Authors
- Federico Betti
- Fabio Zoccolan
