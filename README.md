# Code for the paper "Estimation of splicing metrics for NMD-sensitive transcripts"

## Jupyter notebooks

- `all_nmd_inactivation_experiments.ipynb` — NMD inactivation data processing
- `psi_and_expression_gtex.ipynb` — Correlations between PSI and expression in GTEx (needed for Figure 4A)
- `encode_qc.ipynb` — ENCODE RBP knockdown data processing for reproducing Figure 4C,D
- `Figures.ipynb` — Reproducing the other figures
- `RF_NMD_features.ipynb` — Reproducing the supplementary figure showing RF feature importance for classifying NMD-sensitive and NMD-evading events

## Auxiliary code

- `helper_functions.py` — Functions used throughout the Jupyter notebooks
- `mm.py` — Mixture model implementation
- `plotgtf.py` — Plotting genomic tracks for Figure 5
- `nmd_features.py` — Functions for extracting sequence features used in `RF_NMD_features.ipynb`
- `figures.tex` — Assembling figures into panels
- `run_all.cmd` — Running all notebooks sequentially to generate the figures

## Example workflow

The example script `example/scripts/get_gammas.py` demonstrates how to calculate splicing metrics and classify AS events into NMD-sensitive and NMD-evading classes.

Given catalogs of AS events, split-read counts, and metadata describing an NMD inactivation experiment, the script:

1. Calculates percent-spliced-in (PSI) values for AS events in each sample.
2. Estimates individual gamma values.
3. Performs mixture model-based classification of AS events as NMD-sensitive or NMD-evading and calculates class-level gamma values.

### Usage

```bash
python example/scripts/get_gammas.py \
    -N NMD_AS_catalog.tsv \
    -n nonNMD_AS_catalog.tsv \
    -m metadata.tsv \
    -j path/to/junctions \
    -s path/to/sites \
    -o path/to/output
```

### Arguments

- `-N`, `--NMD_catalog` — Catalog of AS events of interest.
- `-n`, `--nonNMD_catalog` — Catalog of AS events that do not trigger NMD.
- `-m`, `--meta` — Metadata describing the NMD inactivation experiment.
- `-j`, `--junctions` — Directory containing junction split-read counts.
- `-s`, `--sites` — Directory containing counts of reads covering splice sites (required for intron retention analysis).
- `-o`, `--output` — Output directory for the results.

For a full list of options, run:

```bash
python example/scripts/get_gammas.py --help
```