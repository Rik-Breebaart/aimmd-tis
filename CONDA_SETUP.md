# AIMMD-TIS Conda Environment Setup

This guide sets up one conda environment with aimmdTIS and the packages it builds on:
[ops-setup](https://github.com/Rik-Breebaart/ops-setup), [aimmd](https://github.com/bio-phys/aimmd)
and [rpe_mbar](https://github.com/Rik-Breebaart/rpe_mbar).

## Install

Clone the four repositories into one parent folder:

```bash
git clone https://github.com/Rik-Breebaart/aimmd-tis
git clone https://github.com/Rik-Breebaart/ops-setup
git clone https://github.com/Rik-Breebaart/rpe_mbar
git clone https://github.com/bio-phys/aimmd
git -C aimmd checkout 1848a7a
```

Create the environment and install the packages in this order:

```bash
conda env create -f aimmd-tis/environment.yml
conda activate aimmd-tis

pip install -e "./ops-setup[openmm]"           # system set-up, engines, descriptor storage
pip install --no-build-isolation -e ./aimmd    # committor models and selectors
pip install -e ./rpe_mbar                      # RPE / multiset MBAR reweighting
pip install -e ./aimmd-tis/src                 # aimmdTIS (setup.py lives in src/)
```

Check the installation:

```bash
cd aimmd-tis/examples/toy_examples
python -c "import aimmd, aimmdTIS, ops_setup, rpe_mbar; print('ok')"
pytest tests              # 4 smoke tests, about 30 s
```

Then run the toy workflow (see [examples/README.md](examples/README.md)):

```bash
python toy_tps_aimmd.py 50 -o results/                     # AIMMD-TPS, about 30 s
python toy_tis_aimmd.py results/ -i -2 -1 -n 20 -d forward  # AIMMD-TIS
python toy_tis_aimmd.py results/ -i 2 1 -n 20 -d backward
```

## Why the versions are pinned

| Pin | Reason |
|---|---|
| aimmd at commit `1848a7a` (0.9) | aimmd 0.10 removed the `cache_file` argument of `PytorchRCModel`, which aimmdTIS passes. aimmd 0.10 also requires OpenPathSampling ≥ 1.6.1, and those releases require NumPy < 2 while current openmmtools requires NumPy ≥ 2. |
| `asyncmd==0.3.3` | aimmd 0.9 imports `construct_TP_from_plus_and_minus_traj_segments`, which asyncmd ≥ 0.4 renamed. aimmd does not pin asyncmd itself, so pip would install the latest version and `import aimmd` would fail. |
| `numpy<2` | aimmd 0.9 writes storage attributes with `np.string_`, which NumPy 2 removed. With NumPy 2, `aimmd.Storage(..., "w")` fails. |
| `openpathsampling=1.6.0` | The version tested with this combination. |
| `mdtraj=1.11.0` | mdtraj 1.11.1 declares `numpy~=2.0` in its pip metadata. conda installs it next to NumPy 1.26 anyway, but the next `pip install -e` of a package that depends on mdtraj (ops-setup) then upgrades NumPy to 2.x. |
| `setuptools<81` | aimmd 0.9 imports `pkg_resources` (in `aimmd/base/storage.py`), which newer setuptools releases removed. |
| `pip install --no-build-isolation` for aimmd | aimmd 0.9 has no `pyproject.toml`, so pip builds it in an isolated environment without Cython and the build fails. Without isolation it uses the Cython from the conda environment. |

`aimmd` is not on conda-forge, so it is not part of `environment.yml`.

## Troubleshooting

**`ImportError: cannot import name 'TrainSet' from 'aimmd' (unknown location)`.**
Python found the `aimmd/` clone folder in the current directory instead of the installed package.
Run Python from a different folder, for example `aimmd-tis/examples/toy_examples`.

**`ImportError: cannot import name 'construct_TP_from_plus_and_minus_traj_segments'`.**
asyncmd is too new: `pip install "asyncmd==0.3.3"`.

**`AttributeError: np.string_ was removed in the NumPy 2.0 release`.**
NumPy 2 is installed. Check `python -c "import numpy; print(numpy.__version__)"`. The usual cause is that pip upgraded NumPy
while installing a package (see the `mdtraj` pin above). Recreate the environment from `environment.yml`.

**`ModuleNotFoundError: No module named 'pkg_resources'`.**
setuptools is too new for aimmd 0.9: `conda install -c conda-forge "setuptools<81"`.

**`ModuleNotFoundError: Cython not found` while installing aimmd.**
Install aimmd with `pip install --no-build-isolation -e ./aimmd`.

**`TypeError: PytorchRCModel.__init__() got an unexpected keyword argument 'cache_file'`.**
aimmd 0.10 is installed. Check out commit `1848a7a` in the aimmd clone and reinstall it.

**OpenMM platforms.** Check which platforms OpenMM can use with
`python -c "from openmm import Platform as P; print([P.getPlatform(i).getName() for i in range(P.getNumPlatforms())])"`.
The toy examples use the OPS toy engine and do not need OpenMM. The host-guest scripts need a
CUDA or OpenCL platform for reasonable speed.

**GPU PyTorch (Linux).** `environment.yml` installs the conda-forge `pytorch` build, which picks a CUDA
build when a GPU driver is present. Check with `python -c "import torch; print(torch.cuda.is_available())"`.

## Updating

```bash
conda env update -f aimmd-tis/environment.yml --prune
git -C aimmd-tis pull && git -C ops-setup pull && git -C rpe_mbar pull
```

The packages are installed in editable mode (`-e`), so pulled changes are used without reinstalling.
Keep the aimmd clone on commit `1848a7a`.

## Remove

```bash
conda deactivate
conda env remove -n aimmd-tis
```
