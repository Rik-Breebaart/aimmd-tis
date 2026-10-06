# Host–guest committor model analysis

Analysis notebooks for the trained host–guest committor model in:

> R. S. Breebaart and P. G. Bolhuis, "Training, validation, and interpretation of
> committor models based on reweighted path ensembles",
> J. Chem. Phys. 165, 124125 (2026). https://doi.org/10.1063/5.0350496

The model is trained on the reweighted path ensemble (RPE) from AIMMD-TIS
simulations of the host–guest system, using 7 descriptors (host–guest distance,
guest orientation and position angles, hydrogen bonds, waters in the cavity,
hydrophobic contact score, shared waters).

## Notebooks

| Notebook | Content | Figures (`Figures/`) |
|---|---|---|
| `committor_validation.ipynb` | Stable-state separation, calibration against shooting outcomes and committor-analysis frames, per-frame loss, path crossing probabilities | `HG_overview_q_chevron`, `HG_committor_validation_along_q`, `HG_committor_calibration`, `HG_committor_validation_committor_frames`, `HG_committor_loss_diagnostics`, `HG_crossing_probability_validation*`, `HG_validation_overview` |
| `committor_descriptor_importance.ipynb` | Gradient, permutation, conditional-permutation and HIPR descriptor importance along q | `HG_gradient_along_q`, `HG_gradient_unit_along_q`, `HG_gradient_magnitude_and_unit_along_q`, `HG_descriptor_importance_summary` |
| `committor_correlation_analysis.ipynb` | RPE-weighted descriptor correlations compared with relations learned by the model | `HG_correlation_matrix`, `HG_correlation_plot_top_pair`, `HG_permutation_influence_correlation`, `HG_descriptor_swap_loss_matrix` |
| `committor_mechanistic_insight.ipynb` | RPE free energy, shooting outcomes, model committor and descriptor distributions projected on descriptor pairs | `HG_free_energy_2d*`, `HG_q_committor_shots*`, `HG_q_model_projection_*`, `HG_projections_and_slices`, `HG_mechanistic_four_projection_summary` |

`committor_analysis_helpers.py` contains the numerical and plotting helpers
shared by the notebooks: weighted losses, correlations, and permutation and
gradient importance. Each notebook does its own imports, path set-up, data and
model loading and `SystemVisualizer` construction, so each notebook runs on its
own.

`committor_descriptor_importance` and `committor_mechanistic_insight` leave out
clearly unbound frames (host–guest distance > 1.1 nm, `clear_unbound = True`).
The other two notebooks use all frames.

## Data

The notebooks expect these files in `Files/`:

| File | Content |
|---|---|
| `trainset_rpe.pkl` | RPE trainset: dict with `descriptors` (N×7), `weights` (N), `shot_results` (N×2) |
| `aimmd_store_s5em04.h5` | aimmd storage with the trained model (key `stage1_full_model_model_best`) |
| `committor_analysis_all_frames.csv`, `committor_analysis_descriptors_all_frames.npy` | Committor-analysis shots and descriptors used for validation (only `committor_validation`) |

These files are not in the repository because of their size (the trainset is
about 450 MB), and `Files/` is git-ignored.

## Running

Use an environment with `aimmd` and `aimmdTIS` installed (see the top-level
README). Start Jupyter in this folder and run a notebook from top to bottom. To
keep the data elsewhere, set `HG_COMMITTOR_WORK_DIR` to a folder that contains
`Files/` before starting Jupyter.

The model output q for all frames is cached in `Files/.cache/`:
`q_model_mbar.npy` for all frames and `q_model_mbar_clear_unbound.npy` with the
unbound frames removed. The gradient statistics are cached in
`gradient_stats_variable_bins_v2.npz`. Set `RECOMPUTE_Q_MODEL = True` (or
`Recompute = True` for the gradients) after changing the model, the trainset or
the bins.
