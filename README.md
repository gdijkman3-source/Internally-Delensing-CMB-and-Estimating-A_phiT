# CMB Lensing Analysis

This project computes and bins CMB lensing cross- and auto-spectra (phi–T and phi–phi) using FFP10 simulations.

## Quick Start (Python 3.10 required)

This project ships compiled extension artifacts built against CPython 3.10 (see `*.cpython-310-*.so`), so you must use Python 3.10.x.

```bash
git clone https://github.com/gdijkman3-source/Internally-Delensing-CMB-and-Estimating-A_phiT.git
cd Internally-Delensing-CMB-and-Estimating-A_phiT

# Ensure a Python 3.10 interpreter is available (examples):
#   Ubuntu 22.04+: sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv
#   pyenv:         pyenv install 3.10.14 && pyenv local 3.10.14

python3.10 -m venv venv
source venv/bin/activate
python --version  # should report 3.10.x
pip install --upgrade pip
pip install -r requirements.txt
```

## Dependencies

The minimal runtime requirements (with critical pins) are in `requirements.txt`:

- numpy==1.26.4 (required build ABI for included extensions), scipy (array computing, stats; pick a version compatible with numpy 1.26.4)
- healpy (HEALPix spherical maps)
- lenspyx (lensing remapping / Wigner operations)
- matplotlib, seaborn (plotting)
- tqdm

Additional (optional) packages you may want for development / notebooks: `ipykernel`, `notebook`, `jupyterlab`.

Install extras if needed (inside the activated Python 3.10 venv):
```bash
pip install ipykernel notebook jupyterlab
python -m ipykernel install --user --name cmb-lensing
```

## Environment Configuration

Edit or create `env_config.py` to define paths (example values shown):
```python
import os

PLENS = os.environ.get("PLENS", "/absolute/path/to/output/")
INPUT = os.environ.get("INPUT", "/absolute/path/to/input/")
PARAMS = os.environ.get("PARAMS", "./input")  # contains mask.fits.gz, dcl_sim, dcl_dat
KFIELD = os.environ.get("KFIELD", "/absolute/path/to/kappa_fields/")
```

Meaning of variables:
- PLENS: Output directory root (will store generated spectra / intermediates)
- INPUT: Directory with simulation CMB maps `cmb_%05d.fits`, noise maps `noise_%05d.fits`, and data map `SMICA.fits`
- PARAMS: Directory holding mask and dCl spectrum inputs (already includes small sample files here)
- KFIELD: Directory for lensing kappa `klm_%03d.fits` inputs if used

You can also export these as environment variables before running scripts instead of editing the file.

## Data Layout (expected)

```
<INPUT>/cmb_00000.fits
<INPUT>/cmb_00001.fits
... (CMB sims)
<INPUT>/noise_00000.fits
<INPUT>/noise_00001.fits
... (noise sims)
<KFIELD>/klm_000.fits
<KFIELD>/klm_001.fits
... (input lensing)
<INPUT>/SMICA.fits           # data map
<PARAMS>/mask.fits.gz        # analysis mask
<PARAMS>/dcl_sim             # sim power adjustment
<PARAMS>/dcl_dat             # data power adjustment
```

## Running the Pipeline

Generate spectra and intermediate products:
```bash
source venv/bin/activate
python run_parfiles.py
```

You can customize which "parameter set" to run by editing imports inside `run_parfiles.py` or the modules under `parfiles/`.

## Notebooks

After generating results, explore the notebooks in `THESIS/`:
- `Amplitude.ipynb` – Lensing amplitude vs. simulations
- `Carron_2017.ipynb` – Reproduction of literature results
- `PP_results.ipynb` – Phi–phi auto-spectrum
- `PT_results.ipynb` – Phi–T cross-spectrum