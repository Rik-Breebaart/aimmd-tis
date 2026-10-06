# AIMMD-TIS-Framework

This repository contains the code and resources for the AIMMD-TIS method from the papers:

- "Understanding Mechanisms of Molecular Rare Events from Start to Finish". R. S. Breebaart, G. Lazzeri, R. Covino, and P. G. Bolhuis, Phys. Rev. Lett. 136, 168001 (2026). https://doi.org/10.1103/lk32-njx7
- "Training, validation, and interpretation of committor models based on reweighted path ensembles". R. S. Breebaart and P. G. Bolhuis, J. Chem. Phys. 165, 124125 (2026). https://doi.org/10.1063/5.0350496

AIMMD-TIS combines [AIMMD](https://github.com/bio-phys/aimmd), which learns the committor from shooting data, with Transition Interface Sampling in [OpenPathSampling](https://openpathsampling.org). The learned committor is used as the TIS order parameter. The reweighted path ensemble from TIS then gives better training data for the next committor model.

## What this repository includes

- **`aimmdTIS` package** (`src/aimmdTIS/`): an extension of AIMMD for TIS.
  - `AIMMDSetup`: builds the committor model and shooting-point selector from a config dict.
  - `AIMMD_TIS`: runs AIMMD-TIS interface by interface, with the model committor as order parameter.
  - `rcmodel`, `selector`, `stochastic_gates`: TIS-specific committor model, selector and descriptor gates.
  - `training`: training loops and losses for (reweighted) TIS data.
  - `diagnostics`: interface placement and committor validation.
  - `visualization`: model analysis plots (`SystemVisualizer`, `ToyVisualizer`).
  - `openmm_setup`: optional wrappers around the ops-setup OpenMM host-guest set-up.
- **Examples** (`examples/`): the workflow on a toy potential (notebook and scripts) and OpenMM run scripts. See [examples/README.md](examples/README.md).
- **Analysis notebooks** (`examples/Analysis_examples/`): the committor-model training, validation and interpretation analysis from the J. Chem. Phys. paper.

## Dependencies

- [aimmd](https://github.com/bio-phys/aimmd), [OpenPathSampling](https://openpathsampling.org), PyTorch.
- [ops-setup](https://github.com/Rik-Breebaart/ops-setup): system set-up (toy potentials, OpenMM host-guest), descriptor storage and trajectory loading. The examples use it throughout. The core `aimmdTIS` modules import without it, so the model, selector and `AIMMD_TIS` can also be used with your own OPS engines and states. Only `aimmdTIS.openmm_setup` and the descriptor-storage option of `AIMMD_TIS` need ops-setup.
- [rpe_mbar](https://github.com/Rik-Breebaart/rpe_mbar): computes the reweighted path ensemble (RPE) and crossing probabilities from TIS data. It combines several interface sets (e.g. forward and backward TIS) with multiset MBAR, following R. S. Breebaart and P. G. Bolhuis, "Combining multiple interface set path ensembles with MBAR reweighting", J. Chem. Phys. 164, 104116 (2026). https://doi.org/10.1063/5.0318283. rpe_mbar is a separate package and is not included here. It is needed for the reweighting step and is used in the example notebooks.

## Installation

```bash
conda env create -f environment.yml
conda activate aimmd-tis

pip install -e /path/to/aimmd
pip install -e /path/to/ops-setup
pip install -e /path/to/rpe_mbar
pip install -e src/          # aimmdTIS (setup.py lives in src/)

python -c "import aimmd, aimmdTIS, ops_setup; print('ok')"
```

See [CONDA_SETUP.md](CONDA_SETUP.md) for more detail.

## Quick start

```bash
cd examples/toy_examples
python toy_tps_aimmd.py 100 -o results/            # AIMMD-TPS, trains the committor
python toy_tis_aimmd.py results/ -i -3 -2 -1 -n 50 # AIMMD-TIS on the learned committor
```

For the full workflow, including stable-state interface placement, RPE reweighting and training, open `examples/aimmd_with_toy_systems.ipynb`.

## Reference

If you use this repository, please cite the papers and the upstream AIMMD and OPS projects. If you use the RPE/MBAR reweighting, please also cite the MBAR paper:

```bibtex
@article{breebaart2026aimmdtis,
  title   = {Understanding Mechanisms of Molecular Rare Events from Start to Finish},
  author  = {Breebaart, R. S. and Lazzeri, G. and Covino, R. and Bolhuis, P. G.},
  journal = {Phys. Rev. Lett.},
  volume  = {136},
  pages   = {168001},
  year    = {2026},
  doi     = {10.1103/lk32-njx7}
}

@article{breebaart2026committor,
  title   = {Training, validation, and interpretation of committor models based on reweighted path ensembles},
  author  = {Breebaart, Rik S. and Bolhuis, Peter G.},
  journal = {J. Chem. Phys.},
  volume  = {165},
  number  = {12},
  pages   = {124125},
  year    = {2026},
  doi     = {10.1063/5.0350496}
}

@article{breebaart2026mbar,
  title   = {Combining multiple interface set path ensembles with MBAR reweighting},
  author  = {Breebaart, Rik S. and Bolhuis, Peter G.},
  journal = {J. Chem. Phys.},
  volume  = {164},
  number  = {10},
  pages   = {104116},
  year    = {2026},
  doi     = {10.1063/5.0318283}
}
```
