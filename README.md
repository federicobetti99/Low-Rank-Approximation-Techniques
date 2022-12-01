# Low Rank Approximation Techniques
This repository contains the implememention side of the leverage score sampling project, carried out in the context of the MATH-403 Low-Rank Approximation Techniques course at EPFL.

## Repository description
- `code` - Implementation of the experiments
  - `utils` - Subfolder with utilities functions
  - `run.m` - Main script to run to reproduce comparison between sampling strategies
  - `lambda_tuning.m` - Secondary script to run to reproduce adaptive regularization parameter experiments
  - `halving.m` - Secondary script to run to reproduce repeated halving experiments to estimate leverage scores
- `report` - Folder containing the report of the obtained results
  - `report.pdf` - Report of the obtained results in pdf format
  
## Reproducibility of the results
We provide the interested reader of our report with Matlab scripts to reproduce the plots shown in the report. The script `run.m` carries out a comparison between uniform sampling, columns norm sampling and ridge leverage scores sampling for a Hilbert matrix of size 100, and produces a plot of the average projection error for all the strategies. The script `lambda_tuning.m` shows that for a wise choice of the regularization parameter, which we specified in the report, ridge leverage scores are basically equivalent to sampling from the rows of the k-truncated matrix of the right singular vectors of A. Finally, the script `halving.m` implements the repeated halving algorithm and compares its performance to ridge leverage scores, thus reproducing the last plot shown in the report. 
  
## Authors
- Federico Betti
- Fabio Zoccolan
