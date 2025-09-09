# CMB Lensing Analysis

This project computes and bins CMB lensing cross- and auto-spectra (phi-T and phi-phi) using FFP10 simulations.

## Setup

1. Activate the existing virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Ensure the central configuration file `env_config.py` is present at the project root. It defines the following environment variables:
   - `PLENS`: Path to an (initially empty) directory where all output results will be written.
   - `INPUT`: Path to the folder containing input simulations:
     - CMB maps named `cmb_%05d.fits` (zero-padded indices).
     - Noise maps named `noise_%05d.fits`.
     - Data map saved as `SMICA.fits`.
   - `PARAMS`: Path to parameter inputs (masks, dcl files). Already set by default to `./input`.
   - `KFIELD`: Path that will hold input lensing kappa fields (harmonic coefficients `klm` files).

## Data Layout

- `<INPUT>/cmb_00000.fits`, `<INPUT>/cmb_00001.fits`, …  (unlensed CMB simulations)
- `<INPUT>/noise_00000.fits`, `<INPUT>/noise_00001.fits`, …  (noise simulations)
- `<INPUT>/SMICA.fits`  (observed data map)
- `<PARAMS>/mask.fits.gz`  (analysis mask)
- `<PARAMS>/dcl_sim`, `<PARAMS>/dcl_dat`  (input power spectra files)

## Usage

Run the parameter‐file pipeline to generate maps and spectra:
```bash
python run_parfiles.py
```

## Next Steps

Once `run_parfiles.py` completes, open the `THESIS/` directory to explore the Jupyter notebooks that generate the analysis plots:

- `Amplitude.ipynb`: Lensing amplitude over simulations and data
- `Carron_2017.ipynb`: Reproducing results from Carron et al. (2017)
- `PP_results.ipynb`: Phi–Phi auto-spectrum results and diagnostics
- `PT_results.ipynb`: Phi–T cross-spectrum results and diagnostics

These notebooks contain the plotting code and narrative to visualize and interpret your results.