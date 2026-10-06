# Legacy examples (reference only)

These scripts and notebooks were used with earlier versions of AIMMD-TIS,
including the runs for the PRL paper (Phys. Rev. Lett. 136, 168001 (2026)).
They are kept so the original workflow can be read, but **they do not run with
the current `aimmdTIS` package**.

- `openmm_run_scripts/`: host-guest OpenMM run and analysis scripts. They import
  the old `aimmd.aimmd` namespace and the old `HostGuest/` helpers.
- `toy_notebooks/`: toy-potential notebooks. They call `aimmdTIS.potential_switch`,
  `CallableVolume` and `potential_0`. The toy potentials now live in
  `ops_setup.systems.examples.toy_systems`.

For the current workflow, see the examples one level up (`../README.md`).
