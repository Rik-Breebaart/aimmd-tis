# AIMMD-TIS Examples

This directory contains examples demonstrating the integration of AIMMD (Automatic Identification of Molecular Dynamics) with TIS using the ops-setup infrastructure.

## Structure

```
examples/
├── aimmd_with_toy_systems.ipynb              # AIMMD learning with toy potentials
├── AIMMD_TIS_openmm_run_scripts/             # OpenMM molecular dynamics examples
├── toy_examples/                              # Pure TIS examples (without AIMMD)
└── README.md                                  # This file
```

## Quick Navigation

### For AIMMD Learning
- **Start here**: `aimmd_with_toy_systems.ipynb`
  - Shows how to use ops-setup TIS infrastructure
  - Train AIMMD model on collected TIS trajectories
  - Validate learned reaction coordinate

### For Molecular Dynamics (OpenMM)
- **See**: `AIMMD_TIS_openmm_run_scripts/`
  - Full molecular dynamics simulations
  - AMBER topology support
  - Real systems (host-guest complexes)

### For Pure TIS (No AIMMD)
- **See**: `toy_examples/` and ops-setup `examples/`
  - Basic TIS workflows without learning
  - Good for understanding the method
  - Faster to run for testing

## AIMMD Workflow

The AIMMD-TIS approach iteratively refines the reaction coordinate:

```
Iteration 1:
  1. Setup TIS with manual order parameter (e.g., x-coordinate)
  2. Run TIS sampling
  3. Collect trajectories from sampling
  4. Train AIMMD neural network
  5. Validate predictions
  
Iteration 2:
  6. Use AIMMD-learned coordinate as new order parameter
  7. Run TIS with improved order parameter
  8. Retrain AIMMD on larger dataset
  9. Measure improved sampling efficiency
  
Iteration N:
  - Converge to optimal reaction coordinate
  - Get accurate transition rate estimates
```

## Example Files

### `aimmd_with_toy_systems.ipynb`

**Purpose**: Demonstrate AIMMD training with toy potentials

**What it shows**:
1. Creating TIS setup using ops-setup
2. Running TIS to generate training data
3. Training PyTorch neural network for reaction coordinate
4. Comparing AIMMD predictions to true coordinates
5. Extracting improved interface values

**Key imports**:
```python
from ops_setup.systems.examples.toy_systems import ToyTISSetup
from aimmdTIS.Training import train_AIMMD_model
import torch  # For neural network training
```

**Expected runtime**: 2-5 minutes

**Output**:
- `toy_aimmd_tis_sampling.nc` - TIS sampling data
- Plots comparing AIMMD vs true coordinate
- Suggested interface values from AIMMD

## Comparison: Pure TIS vs AIMMD-TIS

| Aspect | Pure TIS | AIMMD-TIS |
|--------|----------|-----------|
| Order Parameter | Manual (e.g., distance) | Learned from data |
| Setup Time | Fast | Longer (includes training) |
| First Run Efficiency | Variable | Poor (learning phase) |
| Convergence | Slow without good OP | Faster with iterations |
| Accuracy | Depends on OP choice | Improves each iteration |
| Ideal For | Testing, known systems | Unknown order parameters |

## Key Differences from ops-setup

**ops-setup** (`/code/ops-setup/examples/`):
- Pure TPS/TIS without learning
- Focus on sampling infrastructure
- Good for understanding methods
- Fast to run

**aimmd-tis** (this directory):
- TIS combined with AIMMD learning
- Automatically optimizes order parameter
- More complex but often more efficient
- Uses same ops-setup infrastructure

## Running Examples

### Interactive (Jupyter)
```bash
# Navigate to aimmd-tis/examples/
cd examples/

# Start Jupyter
jupyter notebook

# Open aimmd_with_toy_systems.ipynb
```

### Command Line
```python
# Run in Python script
from ops_setup.systems.examples.toy_systems import ToyTISSetup
import openpathsampling as paths

tis = ToyTISSetup(potential_name='potential_1')
# ... continue with TIS workflow
```

## Common Workflows

### Workflow 1: Learn Reaction Coordinate with AIMMD

```python
from ops_setup.systems.examples.toy_systems import ToyTISSetup
import openpathsampling as paths

# 1. Setup TIS (with initial guess for order parameter)
tis = ToyTISSetup(potential_name='potential_1', interface_values=[0.2, 0.4, 0.6, 0.8])

# 2. Run TIS sampling
# ... (see notebook for details)

# 3. Train AIMMD
# ... (see notebook for training code)

# 4. Use learned coordinate for next iteration
new_interfaces = aimmd_model.predict(coordinates)
```

### Workflow 2: Validate AIMMD on Toy System

```python
# Use simple system to validate AIMMD works correctly
tis = ToyTISSetup(potential_name='potential_1')

# Compare AIMMD prediction to known good order parameter (x-coordinate)
# Can test different neural network architectures
# Can benchmark sampling efficiency improvement
```

### Workflow 3: Prepare for Molecular MD

```python
# Use toy potential AIMMD as proof-of-concept
# Train on toy system shows feasibility
# Apply same approach to molecular systems
# (See AIMMD_TIS_openmm_run_scripts/ for molecular examples)
```

## Dependencies

Required:
- `openpathsampling` - Core TIS/TPS library
- `numpy` - Numerical computing
- `torch` - Neural network training

Optional:
- `matplotlib` - Plotting (for notebooks)
- `openmm` - Molecular dynamics (for OpenMM examples)
- `pytorch` - Advanced AIMMD training

## Tips and Best Practices

### For AIMMD Training
1. **Collect enough data**: More TIS samples → better training
2. **Validate predictions**: Always check correlation with true values
3. **Use appropriate network**: Simple networks (2-3 layers) often work best
4. **Normalize inputs**: Standardize coordinates before training
5. **Monitor convergence**: Track how AIMMD improves over iterations

### For Toy System Testing
1. **Start with potential_0**: Fastest for quick tests
2. **Use potential_1 for validation**: Good balance of complexity
3. **Save trained models**: Reuse for multiple TIS runs
4. **Monitor acceptance rates**: Should be 20-40% for good sampling

### Scaling to Molecular Systems
1. **Master toy systems first**: Understand method without simulation complexity
2. **Use ops-setup infrastructure**: Same classes work for any system
3. **Adapt AIMMD training**: May need different network architecture
4. **Test on small systems**: Host-guest before proteins

## Troubleshooting

### "Module not found" errors
```python
# Ensure ops-setup is installed
pip install -e /path/to/ops-setup
```

### Neural network not converging
- Reduce learning rate: `lr=0.001` instead of `0.01`
- Use more training iterations: `epochs=500` instead of `100`
- Check input normalization

### TIS sampling very slow
- Use ops-setup TPS instead (simpler)
- Check interface values are reasonable
- Ensure initial trajectory is valid

## Example Output

Running `aimmd_with_toy_systems.ipynb` produces:
```
System: toy_system_1
Engine: toy_engine
Interfaces (x-coordinate): [0.2, 0.4, 0.6, 0.8]

Creating initial trajectory...
Running TIS for 20 MC steps...
TIS sampling complete!
Total trajectories: 20

Training AIMMD model...
Epoch     Loss
20        0.045234
40        0.023156
...
100       0.001045

Correlation between true and AIMMD coordinate: 0.9852

Interface Value Comparison:
Percentile      True          AIMMD
20%             0.1850        0.1923
40%             0.3920        0.3845
60%             0.6150        0.6234
80%             0.8050        0.7956
```

## Next Steps

1. **Master toy systems** (this directory)
2. **Move to OpenMM examples** (AIMMD_TIS_openmm_run_scripts/)
3. **Apply to real systems** (proteins, complexes)
4. **Optimize AIMMD network** (architecture, training)
5. **Calculate transition rates** (from TIS sampling)

## Documentation

- [ops-setup README](../../code/ops-setup/README.md) - TIS/TPS infrastructure
- [ops-setup QUICKSTART](../../code/ops-setup/QUICKSTART.md) - Getting started
- [AIMMD-TIS Tools.py](../src/aimmdTIS/Tools.py) - Utilities
- [Training.py](../src/aimmdTIS/Training.py) - AIMMD training functions

## File Organization

```
aimmd-tis/
├── src/aimmdTIS/
│   ├── Tools.py                   # Utilities (now uses ops-setup)
│   ├── Training.py                # AIMMD training
│   ├── Toy_potentials.py          # Toy PES (now in ops-setup)
│   └── ...
│
├── examples/
│   ├── aimmd_with_toy_systems.ipynb         # AIMMD learning example
│   ├── AIMMD_TIS_openmm_run_scripts/        # Molecular dynamics examples
│   ├── toy_examples/                        # Pure TIS examples
│   └── README.md                            # This file
│
└── ...
```

## Citation

If you use AIMMD-TIS in your research, please cite:
- OpenPathSampling: [https://openpathsampling.org/](https://openpathsampling.org/)
- AIMMD: [Original publication]

---

**Last Updated**: 2024
**Compatible with**: ops-setup 1.0.0+, OpenPathSampling 1.0+
