# AIMMD-TIS Conda Environment Setup

This guide explains how to set up conda environments for aimmd-tis development and usage.

## Quick Start (2 minutes)

```bash
# Create environment
cd /Users/rbreeba/code/aimmd-tis
conda env create -f environment.yml

# Activate
conda activate aimmd-tis

# Install dependencies
pip install -e /Users/rbreeba/code/ops-setup[openmm]  # ops-setup
pip install -e /Users/rbreeba/code/aimmd              # aimmd
pip install -e /Users/rbreeba/code/aimmd-tis/src      # aimmd-tis (setup.py lives in src)

# Verify
python -c "import aimmdTIS, aimmd, ops_setup; print('✓ Ready!')"
```

---

## Installation Order

**Important**: Install packages in this order:

1. **ops-setup** (foundation - TPS/TIS infrastructure)
2. **aimmd** (machine learning for RCs)
3. **aimmd-tis** (integration layer)

```bash
# Step 1: Activate environment
conda activate aimmd-tis

# Step 2: Install ops-setup
pip install -e /Users/rbreeba/code/ops-setup[openmm]

# Step 3: Install aimmd
pip install -e /Users/rbreeba/code/aimmd

# Step 4: Install aimmd-tis
pip install -e /Users/rbreeba/code/aimmd-tis/src
```

---

## Environment Features

The `environment.yml` includes:

### Base Scientific Stack
- Python 3.10
- NumPy, SciPy, Matplotlib
- Jupyter, IPython

### Molecular Dynamics
- OpenPathSampling (path sampling)
- OpenMM (MD engine)
- OpenMMTools (utilities)
- MDTraj (trajectory analysis)

### Machine Learning
- PyTorch with CUDA 11.8 support
- CPU-only alternative available

### Development Tools
- pytest (testing)
- Jupyter (notebooks)
- Black, Flake8 (code quality)

---

## Installation Methods

### Method 1: From Environment File (Recommended)

```bash
# Create environment
conda env create -f /Users/rbreeba/code/aimmd-tis/environment.yml

# Activate
conda activate aimmd-tis

# Install packages
pip install -e /Users/rbreeba/code/ops-setup[openmm]
pip install -e /Users/rbreeba/code/aimmd
pip install -e /Users/rbreeba/code/aimmd-tis
```

### Method 2: Manual Setup

```bash
# Create Python 3.10 environment
conda create -n aimmd-tis python=3.10 -c conda-forge

# Activate
conda activate aimmd-tis

# Install scientific stack
conda install -c conda-forge \
  numpy scipy matplotlib \
  jupyter ipython \
  pytest pytest-cov \
  openpathsampling \
  openmm openmmtools mdtraj \
  h5py networkx pyyaml

# Install PyTorch (choose one)
# Option A: GPU support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Option B: CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install your packages
pip install -e /Users/rbreeba/code/ops-setup[openmm]
pip install -e /Users/rbreeba/code/aimmd
pip install -e /Users/rbreeba/code/aimmd-tis/src
```

### Method 3: Minimal Environment

```bash
# Lightweight environment - use if storage is limited
conda create -n aimmd-tis-minimal python=3.10 -c conda-forge

conda activate aimmd-tis-minimal

# Install only essential packages
conda install -c conda-forge \
  numpy scipy \
  openpathsampling \
  openmm openmmtools mdtraj \
  pytorch::pytorch pytorch::pytorch-cuda=11.8

pip install -e /Users/rbreeba/code/ops-setup[openmm]
pip install -e /Users/rbreeba/code/aimmd
pip install -e /Users/rbreeba/code/aimmd-tis/src
```

---

## Using the Environment

### Activate for Work

```bash
conda activate aimmd-tis
```

### Check Installed Packages

```bash
conda list
```

### Run Python Script

```bash
python my_script.py
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

### Deactivate

```bash
conda deactivate
```

---

## PyTorch GPU Setup

### Check CUDA Version

```bash
# Check your CUDA installation
nvidia-smi

# Will show: CUDA Version: X.X
```

### Install Correct PyTorch

```bash
# For CUDA 11.8 (most common)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU only (no GPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Verify GPU Access

```bash
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
EOF
```

---

## Troubleshooting

### Import Errors

```bash
# Error: "No module named 'aimmd'"
# Solution: Make sure aimmd is installed

conda activate aimmd-tis
pip install -e /Users/rbreeba/code/aimmd

# Verify
python -c "import aimmd; print('OK')"
```

### OpenMM/CUDA Issues

```bash
# Error: "No OpenMM Platform"
# Solution: Check OpenMM installation

python << 'EOF'
from openmm import Platform
print("Available platforms:", Platform.getPlatformNames())
EOF

# If only CPU appears, reinstall with GPU support
conda install -c conda-forge openmm

# Or try:
conda install openmm::openmm -c conda-forge
```

### PyTorch GPU Not Detected

```bash
# Check GPU availability
python << 'EOF'
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("GPU not found - check nvidia-smi and CUDA installation")
EOF

# Reinstall PyTorch with correct CUDA version
conda remove pytorch torchvision torchaudio pytorch-cuda -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Memory Issues

```bash
# If environment creation fails due to memory:
# 1. Clear conda cache
conda clean --all -y

# 2. Create minimal environment first
conda create -n aimmd-tis-min python=3.10 -c conda-forge

# 3. Install packages one at a time
conda activate aimmd-tis-min
conda install -c conda-forge numpy -y
conda install -c conda-forge scipy -y
# ... continue for other packages
```

### Conflicting Dependencies

```bash
# If you get dependency conflicts, try:
# 1. Create fresh environment
conda env remove -n aimmd-tis
conda env create -f environment.yml

# 2. Or manually resolve
conda install -c conda-forge --update-all
```

---

## Development Setup

For active development of aimmd-tis:

```bash
# Create development environment
conda env create -f /Users/rbreeba/code/aimmd-tis/environment.yml -n aimmd-tis-dev

# Activate
conda activate aimmd-tis-dev

# Install all three packages in editable mode
pip install -e /Users/rbreeba/code/ops-setup[openmm,dev]
pip install -e /Users/rbreeba/code/aimmd[dev]
pip install -e /Users/rbreeba/code/aimmd-tis/src

# Install additional dev tools
conda install -c conda-forge black flake8 mypy isort
```

### Running Tests

```bash
conda activate aimmd-tis-dev

# From aimmd-tis directory
cd /Users/rbreeba/code/aimmd-tis
pytest tests/

# With coverage
pytest --cov=aimmdTIS tests/
```

### Code Quality Checks

```bash
# Format code
black /Users/rbreeba/code/aimmd-tis/src

# Check style
flake8 /Users/rbreeba/code/aimmd-tis/src

# Type checking
mypy /Users/rbreeba/code/aimmd-tis/src
```

---

## Multiple Environments

Use different environments for different purposes:

```bash
# Production use
conda env create -f environment.yml -n aimmd-tis

# Development
conda env create -f environment.yml -n aimmd-tis-dev

# Testing
conda env create -f environment.yml -n aimmd-tis-test

# Minimal
conda create -n aimmd-tis-min python=3.10 -c conda-forge

# Specific version testing
conda create -n aimmd-tis-torch2 python=3.10 -c conda-forge
# Then install torch 2.0 specifically
```

Switch between them:

```bash
conda activate aimmd-tis
# or
conda activate aimmd-tis-dev
# or
conda activate aimmd-tis-test
```

---

## Updating Packages

### Update Environment

```bash
# Update from environment file
conda env update -f environment.yml --prune

# Update all packages
conda update --all -c conda-forge

# Update specific package
conda install -c conda-forge numpy --update-all
```

### Update Installed Packages

```bash
# Update ops-setup
pip install --upgrade /Users/rbreeba/code/ops-setup[openmm]

# Update aimmd
pip install --upgrade /Users/rbreeba/code/aimmd

# Update aimmd-tis
pip install --upgrade /Users/rbreeba/code/aimmd-tis
```

---

## Backup and Recovery

### Export Environment

```bash
# Create backup
conda env export -n aimmd-tis > aimmd-tis-backup-$(date +%Y%m%d).yml

# For sharing
conda env export -n aimmd-tis --no-builds > aimmd-tis-nobuilds.yml
```

### Restore from Backup

```bash
# Create environment from backup
conda env create -f aimmd-tis-backup-20260128.yml

# Or update existing
conda env update -f aimmd-tis-backup-20260128.yml
```

---

## Environment Variables

Set environment variables for the conda environment:

```bash
# Create activation script
mkdir -p /Users/rbreeba/miniconda3/envs/aimmd-tis/etc/conda/activate.d
mkdir -p /Users/rbreeba/miniconda3/envs/aimmd-tis/etc/conda/deactivate.d

# Create activation script
cat > /Users/rbreeba/miniconda3/envs/aimmd-tis/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export OPS_SETUP_PATH=/Users/rbreeba/code/ops-setup
export AIMMD_PATH=/Users/rbreeba/code/aimmd
export AIMMD_TIS_PATH=/Users/rbreeba/code/aimmd-tis
EOF

# Create deactivation script
cat > /Users/rbreeba/miniconda3/envs/aimmd-tis/etc/conda/deactivate.d/env_vars.sh << 'EOF'
#!/bin/bash
unset OPS_SETUP_PATH
unset AIMMD_PATH
unset AIMMD_TIS_PATH
EOF

chmod +x /Users/rbreeba/miniconda3/envs/aimmd-tis/etc/conda/activate.d/env_vars.sh
chmod +x /Users/rbreeba/miniconda3/envs/aimmd-tis/etc/conda/deactivate.d/env_vars.sh
```

Now these variables will be set when you activate the environment.

---

## Summary

| Task | Command |
|------|---------|
| Create | `conda env create -f environment.yml` |
| Activate | `conda activate aimmd-tis` |
| Install ops-setup | `pip install -e /path/to/ops-setup[openmm]` |
| Install aimmd | `pip install -e /path/to/aimmd` |
| Install aimmd-tis | `pip install -e /path/to/aimmd-tis` |
| Run tests | `pytest tests/` |
| Check GPU | `python -c "import torch; print(torch.cuda.is_available())"` |
| Deactivate | `conda deactivate` |
| Remove | `conda remove -n aimmd-tis --all` |
| Update | `conda env update -f environment.yml --prune` |
| Export | `conda env export > backup.yml` |

---

## Next Steps

1. Create environment: `conda env create -f environment.yml`
2. Activate: `conda activate aimmd-tis`
3. Verify GPU (if available): `nvidia-smi`
4. Install packages in order (ops-setup → aimmd → aimmd-tis)
5. Run verification tests
6. Start using AIMMD-TIS!

For help with specific issues, see the **Troubleshooting** section above.
