# CPC Retrieval Redo

This repository contains a pixel-wise 1D-Var retrieval workflow for Juno MWR polar data, along with the local model builds, observation products, and plotting notebooks used to analyze the results.

The retrieval is driven by two Python entry points:

- `run_1dvar_pixelwise_retrieve_dry.py`: dry-background retrieval using the local `cpc_dry` CANOE/CPC build. The retrieved state vector is `(RHmax, dlnNH3/dlnP, pmax)`.
- `run_1dvar_pixelwise_retrieve_moist.py`: pseudo-moist retrieval using the local `cpc_moist` CANOE/CPC build. The retrieved state vector is `(qH2O, RHmax)`.

Both scripts read the gridded brightness-temperature anomaly file `observation/tb_anomaly_grided_PJ51_PJ61_FA02.h5` (too large thus not tracked), run forward radiative transfer through the local CANOE Python bindings, and write HDF5 output grids of retrieved parameters, simulated TBs, and vertical profiles.

## Repository layout

### Retrieval drivers

- `run_1dvar_pixelwise_retrieve_dry.py`
  Dry retrieval driver. Produces `thetas_in_grid_dry.h5` and `profiles_in_grid_dry.h5`.
- `run_1dvar_pixelwise_retrieve_moist.py`
  Pseudo-moist retrieval driver. Produces `thetas_in_grid_pseudo.h5` and `profile_in_grid_pseudo.h5`.
- `juno_mwr.inp`
  Main model input file used by the retrieval scripts.
- `juno_mwr.yaml`
  Main YAML configuration used by the retrieval scripts.
- `log.ret_dry.txt`, `log.ret_pseudo.txt`
  Runtime logs from example dry and pseudo-moist retrieval runs.

### install CANOE

To run the retrieval, you must install the CANOE (Comprehensive Atmosphere N' Ocean Engine) model to enable planetary atmospheric construction and radiative transfer calculations.
We refer to the adapted version we bind the source C++ code into python interfaces. 

```bash
$ git clone https://github.com/jihenghu/canoe.git
$ git check out jh/juno_mwr_fwd
```
Then, follow the instruction of official Canoe model.

### Plotting and analysis notebooks

- `plot_xsecs/`
  Notebooks and figures for cross-section plots of retrieved quantities.
- `plots_extends/`
  Notebooks for extended parameter and polar-mean visualizations.
- `Yuan_simulation/`
  Supporting MPAS simulation outputs and a notebook for related diagnostics.
- `junocamimage/`
  JunoCam/JIRAM image products and reprojection notebooks used for comparison and context.

## Typical workflow

1. Prepare or update the gridded observation product in `observation/`.
2. Choose a retrieval mode and run one of:

```bash
python run_1dvar_pixelwise_retrieve_dry.py
python run_1dvar_pixelwise_retrieve_moist.py
```

3. Inspect the HDF5 outputs written in the repository root.
4. Use the notebooks in `plot_xsecs/`, `plots_extends/`, `observation/`, or `junocamimage/` to visualize the results.


