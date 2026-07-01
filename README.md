# Code for the paper "Estimation of splicing metrics for NMD-sensitive transcripts"

## Jupyter notebooks

- all_nmd_inactivation_experiments.ipynb --- NMD inactivation data processing
- psi_and_expression_gtex.ipynb --- Correlations between psi and expression in GTEx (needed for Figure 4A)
- encode_qc.ipynb --- ENCODE RBP KD data procesing for reproducing Figure 4C,D
- Figures.ipynb --- reproducing all the other figures

## Auxiliary code

- helper_functions.py --- functions used throughout the jupyter notebooks
- mm.py --- mixture model
- plotgtf.py --- plotting Figure 5 genomic tracks
- figures.tex --- assembling figures into panels
- run_all.cmd --- run all notebooks sequentially to obtain figures