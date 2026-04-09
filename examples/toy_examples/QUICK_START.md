# 🎯 QUICK START - Toy Examples with AIMMD

## What's New?

✨ **Toy TPS/TIS examples now fully integrated with ops-setup + AIMMD**

- `toy_tps_aimmd.py` - TPS with AIMMD enhancement
- `toy_tis_aimmd.py` - TIS with AIMMD enhancement (single interface per run)
- `tests/` - Complete testing suite for quick validation

## 🚀 Quick Start (2 minutes)

```bash
# Navigate to toy examples
cd /Users/rbreeba/code/aimmd-tis/examples/toy_examples

# Run quick test validation (1-2 min)
cd tests
python test_toy_systems.py --quick

# Expected: ✓ All tests passed!
```

## 📋 What You Get

### Toy Example Scripts (Production-Ready)

**toy_tps_aimmd.py** (330 lines)
```bash
# Run TPS on 2D potential with AIMMD
python toy_tps_aimmd.py potential_1 10 -o my_tps_run.nc

# With pre-trained model
python toy_tps_aimmd.py potential_1 20 -m /path/to/model.pt

# Help
python toy_tps_aimmd.py --help
```

**toy_tis_aimmd.py** (350 lines)
```bash
# Run TIS on 2D potential with AIMMD (single interface)
python toy_tis_aimmd.py potential_1 0.4 10 -o my_tis_run.nc

# Backward direction
python toy_tis_aimmd.py potential_1 0.4 10 -d backward

# Help
python toy_tis_aimmd.py --help
```

### Test Infrastructure (25 Test Assertions)

**test_toy_systems.py** (450 lines)
- Imports: ops_setup, aimmd, aimmdTIS, openpathsampling, torch
- Setup: TPS, TIS, AIMMD configurations
- Execution: TPS and TIS simulations
- Verification: Storage creation, statistics

**test_integration.py** (110 lines)
- Script import checks
- Subprocess validation
- Quick sanity checks

## 🧪 Test Options

```bash
cd /Users/rbreeba/code/aimmd-tis/examples/toy_examples/tests

# Quick tests (2 steps, ~1-2 min)
python test_toy_systems.py --quick

# Full tests (5 steps, ~2-5 min)
python test_toy_systems.py

# Import-only tests (30 sec)
python test_toy_systems.py --skip-setup --skip-run

# Integration tests
python test_integration.py

# Help
python test_toy_systems.py --help
```

## 📊 What's Tested

```
test_toy_systems.py runs 25 assertions:

✓ Imports (5)
  - ops_setup toy systems
  - aimmd package
  - aimmdTIS package
  - openpathsampling
  - torch (with device info)

✓ TPS Setup (4)
  - ToyTPSSetup creation
  - Trajectory generation
  - Network creation
  - Move scheme creation

✓ TIS Setup (4)
  - ToyTISSetup creation
  - Trajectory generation
  - Network creation
  - Move scheme creation

✓ AIMMD Setup (4)
  - AIMMDSetup initialization
  - RC model creation
  - Selector creation
  - Configuration handling

✓ TPS Run (4)
  - Setup & initialization
  - Simulation execution
  - Storage creation
  - Statistics calculation

✓ TIS Run (4)
  - Setup & initialization
  - Simulation execution
  - Storage creation
  - Statistics calculation
```

## 🏗️ Architecture

```
toy_examples/
├── toy_tps_aimmd.py          ← Uses ToyTPSSetup + AIMMDSetup
├── toy_tis_aimmd.py          ← Uses ToyTISSetup + AIMMDSetup
├── tests/
│   ├── test_toy_systems.py   ← Comprehensive test suite (450 lines)
│   ├── test_integration.py   ← Script validation (110 lines)
│   ├── README.md             ← Complete guide (210 lines)
│   └── TEST_INFRASTRUCTURE_COMPLETE.md
└── [existing notebooks & resources]
```

## 🔧 Integration

**Imports from ops-setup:**
- ToyTPSSetup
- ToyTISSetup
- Toy potentials (potential_0, 1, 2)

**Imports from aimmdTIS:**
- AIMMDSetup (with 13 activation functions)
- load_initial_trajectory
- Storage classes

**Imports from standard libraries:**
- OpenMM (MD engine)
- OpenPathSampling (sampling)
- PyTorch (neural networks)

## ✅ What's Different from Before

### Before
- Custom setup utilities
- Limited activation functions (2)
- No device handling
- Manual storage management
- No automated testing

### After ✨
- Uses ops-setup (clean separation of concerns)
- Enhanced activation functions (13 options)
- Automatic device handling (CUDA/MPS/CPU)
- Integrated storage with AIMMD
- Comprehensive test suite (25 assertions)
- Production-ready CLI
- Complete documentation

## 📚 Documentation

| File | Lines | Purpose |
|------|-------|---------|
| toy_tps_aimmd.py | 330 | TPS + AIMMD script |
| toy_tis_aimmd.py | 350 | TIS + AIMMD script |
| tests/test_toy_systems.py | 450 | Main test suite |
| tests/test_integration.py | 110 | Integration tests |
| tests/README.md | 210 | Testing guide |
| VERIFICATION_CHECKLIST.md | - | Checklist (this folder) |
| TEST_INFRASTRUCTURE_COMPLETE.md | - | Technical details (tests folder) |
| AIMMD_TIS_TOY_EXAMPLES_COMPLETE.md | - | Full overview (root) |

## 🎯 Next Steps

### 1. Validate Everything (1-2 min)
```bash
cd tests
python test_toy_systems.py --quick
```

### 2. Run a Toy Example (5+ min)
```bash
cd ..
python toy_tps_aimmd.py potential_1 20 -o test_tps.nc
python toy_tis_aimmd.py potential_1 0.4 20 -o test_tis.nc
```

### 3. Analyze Results
```bash
python
import openpathsampling as paths
storage = paths.Storage('test_tps.nc', 'r')
print(f"Steps: {len(storage.steps)}")
print(f"Acceptance: {storage.move_scheme.acceptance_rate:.1%}")
```

## 🚨 Troubleshooting

### "ModuleNotFoundError: ops_setup"
```bash
pip install -e /Users/rbreeba/code/ops-setup
```

### "ModuleNotFoundError: aimmdTIS"
```bash
cd /Users/rbreeba/code/aimmd-tis/src
pip install -e .
```

### "CUDA is not available"
Tests automatically use CPU - this is fine!

### Tests timeout
Run with `--skip-run` to skip simulations:
```bash
python test_toy_systems.py --skip-run
```

## 📞 Support

- See `tests/README.md` for detailed troubleshooting
- See `TEST_INFRASTRUCTURE_COMPLETE.md` for technical details
- See `AIMMD_TIS_TOY_EXAMPLES_COMPLETE.md` for full overview
- Scripts have `--help` for usage: `python toy_tps_aimmd.py --help`

## ✨ Key Features

✅ Production-ready toy examples with AIMMD
✅ Clean ops-setup integration
✅ 25 test assertions for validation
✅ Quick (1-2 min) verification possible
✅ Complete documentation
✅ CI/CD ready
✅ Device auto-detection
✅ 13 activation functions supported
✅ Single interface per TIS run (as specified)
✅ Pre-trained model loading support

---

**Status:** ✅ Ready for Use

Start with: `cd tests && python test_toy_systems.py --quick`
