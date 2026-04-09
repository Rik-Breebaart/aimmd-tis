# Toy Systems Testing Guide

This directory contains automated tests for the toy TPS/TIS examples with AIMMD integration.

## Quick Start

### Run All Tests
```bash
# Full test suite
python test_toy_systems.py

# Quick tests (2 steps instead of 5)
python test_toy_systems.py --quick

# Show detailed output
python test_toy_systems.py --verbose
```

### Run Specific Tests
```bash
# Import tests only
python test_toy_systems.py --skip-setup --skip-run

# Setup tests only
python test_toy_systems.py --skip-import --skip-run

# Simulation tests only
python test_toy_systems.py --skip-import --skip-setup
```

### Integration Tests
```bash
# Test that toy scripts can be imported and executed
python test_integration.py
```

## Test Structure

### test_toy_systems.py

Main test suite covering:

1. **Import Tests** - Verify all dependencies are available
   - ops_setup.toy_systems
   - aimmdTIS
   - openpathsampling
   - torch
   - aimmd

2. **Setup Tests** - Verify core functionality
   - TPS setup with ToyTPSSetup
   - TIS setup with ToyTISSetup
   - AIMMD setup and model creation
   - Network and move scheme creation

3. **Run Tests** - Verify simulation execution
   - TPS simulation (minimal steps)
   - TIS simulation (minimal steps)

### test_integration.py

Integration tests verifying:
- toy_tps_aimmd.py script loads without errors
- toy_tis_aimmd.py script loads without errors
- Both scripts have proper CLI help documentation

## Expected Output

### Successful Run
```
======================================================================
  AIMMD-TIS Toy Systems Test Suite
======================================================================

======================================================================
  Testing Imports
======================================================================

→ ops_setup toy systems
  ✓ ops_setup toy systems imported
  ✓ aimmd imported
  ✓ aimmdTIS imported
  ✓ openpathsampling imported
  ✓ torch imported (device: cpu)

...

======================================================================
  Test Summary
======================================================================

Total Passed: 25
Total Failed: 0

✓ All tests passed!
```

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'ops_setup'`

**Solution:** Ensure ops-setup package is installed:
```bash
cd /Users/rbreeba/code/ops-setup
pip install -e .
```

**Error:** `ModuleNotFoundError: No module named 'aimmdTIS'`

**Solution:** Ensure aimmd-tis package is installed:
```bash
cd /Users/rbreeba/code/aimmd-tis/src
pip install -e .
```

### CUDA/Device Issues

**Error:** `RuntimeError: CUDA is not available`

**Solution:** The tests automatically fall back to CPU. Ensure torch is installed:
```bash
pip install torch
```

### Timeout Errors

If tests timeout, try running with smaller step counts:
```bash
python test_toy_systems.py --skip-run  # Skip simulation tests
```

## Running Toy Examples After Tests Pass

Once all tests pass, you can run the full toy examples:

### TPS with AIMMD
```bash
# Minimal run
python ../toy_tps_aimmd.py potential_1 5

# With output file
python ../toy_tps_aimmd.py potential_1 10 -o my_tps_run.nc

# With pre-trained model
python ../toy_tps_aimmd.py potential_1 10 -m /path/to/model.pt
```

### TIS with AIMMD
```bash
# Single interface
python ../toy_tis_aimmd.py potential_1 0.4 5

# With output file
python ../toy_tis_aimmd.py potential_1 0.4 10 -o my_tis_run.nc

# Forward direction
python ../toy_tis_aimmd.py potential_1 0.4 10 -d forward

# Backward direction
python ../toy_tis_aimmd.py potential_1 0.4 10 -d backward
```

## Test Configuration

Tests use minimal parameters for speed:
- Integration runs: 2-5 simulation steps (vs 1000+ in production)
- Temporary directories for storage files
- CPU-only by default (no GPU required)
- Small network sizes (8 hidden units)
- Fast potentials (potential_1)

## Extending Tests

To add new tests, follow the pattern in test_toy_systems.py:

```python
def test_new_feature():
    """Test description."""
    print_header("Testing New Feature")
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        # Your test code here
        print_success("Feature works")
        tests_passed += 1
    except Exception as e:
        print_error(f"Feature failed: {e}")
        tests_failed += 1
    
    return tests_passed, tests_failed
```

Then add to the main() function:
```python
passed, failed = test_new_feature()
total_passed += passed
total_failed += failed
```

## Performance Notes

- Full test suite: ~2-5 minutes (depending on system)
- Quick tests (--quick): ~1-2 minutes
- Import/setup only: ~30 seconds
- Each simulation adds ~30-60 seconds

## CI/CD Integration

For automated testing, use:
```bash
# Exit code 0 if all tests pass
python test_toy_systems.py --quick

# Exit code 1 if any tests fail
```

This can be integrated into GitHub Actions, GitLab CI, or other CI/CD systems.
