# AIMMD-TIS examples

These examples show the current AIMMD-TIS workflow. The simulation set-up
(systems, engines, states, storages) comes from
[ops-setup](https://github.com/Rik-Breebaart/ops-setup). The committor model,
shooting-point selection and AIMMD-TIS sampling come from `aimmdTIS`.
Reweighting of the TIS data uses [rpe_mbar](https://github.com/Rik-Breebaart/rpe_mbar),
a separate package for multiset MBAR (Breebaart and Bolhuis, J. Chem. Phys.
164, 104116 (2026), https://doi.org/10.1063/5.0318283).

```
examples/
├── aimmd_tis_toy_tutorial.ipynb     # guided tutorial: full workflow on a 2D toy potential (start here)
├── aimmd_with_toy_systems.ipynb     # older version of the toy workflow (legacy, see below)
├── config_aimmd.json                # AIMMD model / training settings used by the toy examples
├── wolfe_quapp_iterative/           # iterative AIMMD-TIS (several iterations, multi-set MBAR)
├── toy_examples/                    # the same workflow as command-line scripts
│   ├── toy_tps_aimmd.py             #   step 1: AIMMD-TPS
│   ├── toy_tis_aimmd.py             #   step 2: AIMMD-TIS on the learned committor
│   └── tests/                       #   smoke tests (pytest)
├── AIMMD_TIS_openmm_run_scripts/    # OpenMM (host-guest) run scripts for cluster use
├── Analysis_examples/               # committor-model analysis notebooks (JCP 2026 paper)
└── old/                             # legacy scripts, reference only (do not run with current aimmdTIS)
```

## Workflow

1. **AIMMD-TPS.** Run TPS with an AIMMD selector. A committor model
   `q(x) = log(pB/pA)` is trained on the fly from the shooting outcomes.
2. **Stable-state runs (optional).** Run MD in both states and store the
   descriptors with `ops_setup.descriptor_storage.build_stable_state_storage`.
   `aimmdTIS.diagnostics.check_interfaces` then places the first interfaces on q.
3. **AIMMD-TIS.** `aimmdTIS.AIMMD_TIS.run_TIS_sequentially` runs TIS
   interface by interface, forward (A→B) and backward (B→A), with q as the
   order parameter. Each interface has its own OPS storage, so runs stay short
   and storages stay small. While sampling, each path's descriptors and shot
   result are written to an ops-setup `DescriptorStorage`, and its q extrema to
   a CSV file (row = MC step). The later steps use only these files; the OPS
   storages do not have to be opened again.
4. **Reweighting.** `rpe_mbar` combines the forward and backward interface
   sets with multiset MBAR into a reweighted path ensemble (RPE), and gives
   crossing probabilities. `aimmdTIS.rpe_dataset_from_descriptor_storage`
   builds the `RpeDataset` from the descriptor storages.
5. **Training.** Train the committor on the reweighted data
   (`aimmdTIS.train_one_stage`). The new model defines the interfaces of the
   next iteration.

## Toy potential

### Tutorial notebook

`aimmd_tis_toy_tutorial.ipynb` runs steps 1–5 on the Wolfe-Quapp potential
and explains each step. Start Jupyter from `examples/`. With `QUICK = True`
(default) a run from scratch takes about 10 minutes on a laptop CPU; with
`QUICK = False` it uses the sizes of the reference figures. Each stage saves
its results (in `toy_tutorial_quick/` or `toy_tutorial_results/`) and is
skipped on a rerun; `RERUN = True` redoes everything.

### Legacy notebook

`aimmd_with_toy_systems.ipynb` is the earlier version of the same workflow,
kept for reference. Its `run_*` flags default to `False`, so on a fresh clone
it fails at the first load. Its RPE cell reopens every TIS `.nc` storage, and
the descriptors there are taken one MC step earlier than the MBAR weights.
Use the tutorial notebook instead.

### Iterative example

`wolfe_quapp_iterative/run_iterative.py` repeats the TIS → RPE → training
cycle for several iterations and compares each model with the exact
committor. See `wolfe_quapp_iterative/README.md`.

### Scripts

The scripts run steps 1 and 3 from the command line, on the same toy system
and with the same `config_aimmd.json`:

```bash
cd examples/toy_examples

# Step 1: AIMMD-TPS, writes results/aimmd_tps_wolfe-quapp.h5 and results/tps_wolfe-quapp.nc
python toy_tps_aimmd.py 100 -o results/

# Step 3: AIMMD-TIS on interfaces of the learned q (forward and backward)
python toy_tis_aimmd.py results/ -i -3 -2 -1 -n 50 -d forward
python toy_tis_aimmd.py results/ -i 3 2 1 -n 50 -d backward
```

Run the tests with `pytest tests` from `examples/toy_examples`. They take about
two minutes.

## OpenMM host-guest scripts

`AIMMD_TIS_openmm_run_scripts/` contains the scripts for molecular systems.
They read the system and sampling settings from ops-setup JSON configs, as used
by `aimmdTIS.openmm_setup.TPS_setup` / `TIS_setup`:

- `aimmd_openmm_tps_run.py`: AIMMD-TPS.
- `aimmd_openmm_tis_run.py`: AIMMD-TIS for one interface.
- `aimmd_openmm_parallel_tis.py`: one process per interface.

Run any script with `--help` to see its arguments.

## Analysis notebooks

`Analysis_examples/` contains the notebooks used for the analysis of the trained
committor model in R. S. Breebaart and P. G. Bolhuis, J. Chem. Phys. 165, 124125
(2026), https://doi.org/10.1063/5.0350496. See the README in that folder.

## Legacy examples

`old/` keeps the scripts and notebooks from earlier versions, including those
used for the PRL paper. They show the original workflow but **do not run with
the current `aimmdTIS`**. See `old/README.md`.
