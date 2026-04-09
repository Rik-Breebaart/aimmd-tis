#!/usr/bin/env python
"""
Quick test suite for toy system TPS/TIS with AIMMD.

This script runs minimal simulations to verify all components are working.

Usage:
    python test_toy_systems.py [--quick] [--verbose]

Options:
    --quick    Run minimal tests (default: 2 steps)
    --verbose  Show detailed output
"""

import sys
import argparse
from pathlib import Path
import tempfile
import shutil

# Test utilities
def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_section(text):
    """Print formatted section."""
    print(f"\n→ {text}")


def print_success(text):
    """Print success message."""
    print(f"  ✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"  ✗ {text}")


def test_imports():
    """Test all required imports."""
    print_header("Testing Imports")

    tests_passed = 0
    tests_failed = 0

    # Test 1: OPS setup
    try:
        from ops_setup.systems.examples.toy_systems import ToyTPSSetup, ToyTISSetup
        print_success("ops_setup toy systems imported")
        tests_passed += 1
    except Exception as e:
        print_error(f"ops_setup import failed: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test 2: AIMMD
    try:
        import aimmd
        print_success("aimmd imported")
        tests_passed += 1
    except Exception as e:
        print_error(f"aimmd import failed: {e}")
        tests_failed += 1

    # Test 3: aimmdTIS
    try:
        from aimmdTIS import AIMMDSetup, load_initial_trajectory
        print_success("aimmdTIS imported")
        tests_passed += 1
    except Exception as e:
        print_error(f"aimmdTIS import failed: {e}")
        tests_failed += 1

    # Test 4: OPS
    try:
        import openpathsampling as paths
        print_success("openpathsampling imported")
        tests_passed += 1
    except Exception as e:
        print_error(f"openpathsampling import failed: {e}")
        tests_failed += 1

    # Test 5: torch
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print_success(f"torch imported (device: {device})")
        tests_passed += 1
    except Exception as e:
        print_error(f"torch import failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def test_tps_setup():
    """Test TPS setup with toy systems."""
    print_header("Testing TPS Setup")

    tests_passed = 0
    tests_failed = 0

    try:
        from ops_setup.systems.examples.toy_systems import ToyTPSSetup
        import openpathsampling as paths

        print_section("Creating TPS setup for potential_1")

        tps_setup = ToyTPSSetup(
            potential_name="potential_1",
            integrator_params={"dt": 0.05, "temperature": 1.0, "gamma": 2.5},
        )

        print_success(f"TPS setup created: {tps_setup.system_name}")
        tests_passed += 1

        print_section("Creating initial trajectory")
        paths.PathMover.engine = tps_setup.md_engine
        init_traj = paths.Trajectory(
            tps_setup.pes.simple_initial_path(50, tps_setup.md_engine)
        )
        print_success(f"Initial trajectory created: {len(init_traj)} snapshots")
        tests_passed += 1

        print_section("Creating network and move scheme")
        network = tps_setup.create_network("TPS")
        move_scheme = tps_setup.create_move_scheme(network, "OneWay")
        print_success(f"Network: {type(network).__name__}")
        print_success(f"Move scheme: {type(move_scheme).__name__}")
        tests_passed += 2

    except Exception as e:
        print_error(f"TPS setup test failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


def test_tis_setup():
    """Test TIS setup with toy systems."""
    print_header("Testing TIS Setup")

    tests_passed = 0
    tests_failed = 0

    try:
        from ops_setup.systems.examples.toy_systems import ToyTISSetup
        import openpathsampling as paths

        print_section("Creating TIS setup for potential_1")

        tis_setup = ToyTISSetup(
            potential_name="potential_1",
            integrator_params={"dt": 0.05, "temperature": 1.0, "gamma": 2.5},
            interface_values=[0.4],
        )

        print_success(f"TIS setup created: {tis_setup.system_name}")
        tests_passed += 1

        print_section("Creating initial trajectory")
        paths.PathMover.engine = tis_setup.md_engine
        init_traj = paths.Trajectory(
            tis_setup.pes.simple_initial_path(50, tis_setup.md_engine)
        )
        print_success(f"Initial trajectory created: {len(init_traj)} snapshots")
        tests_passed += 1

        print_section("Creating network and move scheme")
        network = tis_setup.create_network("TIS")
        move_scheme = tis_setup.create_move_scheme(network, "TwoWayShooting")
        print_success(f"Network: {type(network).__name__}")
        print_success(f"Move scheme: {type(move_scheme).__name__}")
        tests_passed += 2

    except Exception as e:
        print_error(f"TIS setup test failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


def test_aimmd_setup():
    """Test AIMMD setup."""
    print_header("Testing AIMMD Setup")

    tests_passed = 0
    tests_failed = 0

    try:
        from aimmdTIS import AIMMDSetup
        from ops_setup.systems.examples.toy_systems import ToyTPSSetup
        import torch

        print_section("Creating ToyTPSSetup")
        tps_setup = ToyTPSSetup(
            potential_name="potential_1",
            integrator_params={"dt": 0.05, "temperature": 1.0, "gamma": 2.5},
        )
        print_success("ToyTPSSetup created")
        tests_passed += 1

        print_section("Creating AIMMDSetup")
        config = {
            "AIMMD_settings": {
                "activation": "ReLU",
                "layers": {"0": 8},
                "dropout": {"0": 0.05},
                "distribution": "lorentzian",
                "scale": 1.0,
                "use_GPU": False,
                "ee_params": {"lr_0": 1e-3},
            }
        }
        aimmd_setup = AIMMDSetup(
            config=config,
            descriptor_dim=2,
            states=(tps_setup.stateA, tps_setup.stateB),
        )
        print_success("AIMMDSetup created")
        tests_passed += 1

        print_section("Creating RC model")
        try:
            import aimmd
            with tempfile.TemporaryDirectory() as tmpdir:
                storage_path = Path(tmpdir) / "test_aimmd.h5"
                aimmd_storage = aimmd.Storage(str(storage_path), "w")
                model = aimmd_setup.setup_RCModel(aimmd_storage)
                print_success("RC model created")
                tests_passed += 1
                aimmd_storage.close()
        except Exception as e:
            print_error(f"RC model creation failed: {e}")
            tests_failed += 1

        print_section("Creating selector")
        try:
            selector = aimmd_setup.setup_selector(model)
            print_success(f"Selector created: {type(selector).__name__}")
            tests_passed += 1
        except Exception as e:
            print_error(f"Selector creation failed: {e}")
            tests_failed += 1

    except Exception as e:
        print_error(f"AIMMD setup test failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


def test_tps_run(n_steps=2):
    """Test TPS run with minimal steps."""
    print_header(f"Testing TPS Run ({n_steps} steps)")

    tests_passed = 0
    tests_failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        try:
            from ops_setup.systems.examples.toy_systems import ToyTPSSetup
            import openpathsampling as paths

            print_section("Creating TPS setup")
            tps_setup = ToyTPSSetup(
                potential_name="potential_1",
                integrator_params={"dt": 0.05, "temperature": 1.0, "gamma": 2.5},
            )
            print_success("TPS setup created")
            tests_passed += 1

            print_section("Preparing simulation")
            paths.PathMover.engine = tps_setup.md_engine
            init_traj = paths.Trajectory(
                tps_setup.pes.simple_initial_path(50, tps_setup.md_engine)
            )
            network = tps_setup.create_network("TPS")
            move_scheme = tps_setup.create_move_scheme(network, "OneWay")
            print_success("Simulation prepared")
            tests_passed += 1

            print_section(f"Running TPS for {n_steps} steps")
            storage_path = tmpdir / "test_tps.nc"
            storage = paths.Storage(str(storage_path), "w", template=tps_setup.template)
            storage.save(tps_setup.md_engine)
            storage.save(move_scheme)
            storage.save(network)

            initial_conditions = move_scheme.initial_conditions_from_trajectories(
                init_traj
            )
            sampler = paths.PathSampling(storage, move_scheme, initial_conditions)
            sampler.run(n_steps)

            print_success(f"TPS run completed: {len(storage.steps)} steps")
            tests_passed += 1

            acceptance = move_scheme.acceptance_rate
            print_success(f"Acceptance rate: {acceptance:.1%}")
            tests_passed += 1

            storage.close()

        except Exception as e:
            print_error(f"TPS run failed: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1

    return tests_passed, tests_failed


def test_tis_run(n_steps=2):
    """Test TIS run with minimal steps."""
    print_header(f"Testing TIS Run ({n_steps} steps)")

    tests_passed = 0
    tests_failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        try:
            from ops_setup.systems.examples.toy_systems import ToyTISSetup
            import openpathsampling as paths

            print_section("Creating TIS setup")
            tis_setup = ToyTISSetup(
                potential_name="potential_1",
                integrator_params={"dt": 0.05, "temperature": 1.0, "gamma": 2.5},
                interface_values=[0.4],
            )
            print_success("TIS setup created")
            tests_passed += 1

            print_section("Preparing simulation")
            paths.PathMover.engine = tis_setup.md_engine
            init_traj = paths.Trajectory(
                tis_setup.pes.simple_initial_path(50, tis_setup.md_engine)
            )
            network = tis_setup.create_network("TIS")
            move_scheme = tis_setup.create_move_scheme(network, "TwoWayShooting")
            print_success("Simulation prepared")
            tests_passed += 1

            print_section(f"Running TIS for {n_steps} steps")
            storage_path = tmpdir / "test_tis.nc"
            storage = paths.Storage(str(storage_path), "w", template=tis_setup.template)
            storage.save(tis_setup.md_engine)
            storage.save(move_scheme)
            storage.save(network)

            initial_conditions = move_scheme.initial_conditions_from_trajectories(
                init_traj
            )
            sampler = paths.PathSampling(storage, move_scheme, initial_conditions)
            sampler.run(n_steps)

            print_success(f"TIS run completed: {len(storage.steps)} steps")
            tests_passed += 1

            acceptance = move_scheme.acceptance_rate
            print_success(f"Acceptance rate: {acceptance:.1%}")
            tests_passed += 1

            storage.close()

        except Exception as e:
            print_error(f"TIS run failed: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1

    return tests_passed, tests_failed


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test toy systems with AIMMD")
    parser.add_argument("--quick", action="store_true", help="Run quick tests (2 steps)")
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--skip-import", action="store_true", help="Skip import tests"
    )
    parser.add_argument("--skip-setup", action="store_true", help="Skip setup tests")
    parser.add_argument("--skip-run", action="store_true", help="Skip run tests")

    args = parser.parse_args()

    n_steps = 2 if args.quick else 5

    total_passed = 0
    total_failed = 0

    print_header("AIMMD-TIS Toy Systems Test Suite")

    # Import tests
    if not args.skip_import:
        passed, failed = test_imports()
        total_passed += passed
        total_failed += failed

    # Setup tests
    if not args.skip_setup:
        passed, failed = test_tps_setup()
        total_passed += passed
        total_failed += failed

        passed, failed = test_tis_setup()
        total_passed += passed
        total_failed += failed

        passed, failed = test_aimmd_setup()
        total_passed += passed
        total_failed += failed

    # Run tests
    if not args.skip_run:
        passed, failed = test_tps_run(n_steps)
        total_passed += passed
        total_failed += failed

        passed, failed = test_tis_run(n_steps)
        total_passed += passed
        total_failed += failed

    # Summary
    print_header("Test Summary")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print()

    if total_failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
