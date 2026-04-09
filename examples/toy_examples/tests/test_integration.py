#!/usr/bin/env python
"""
Integration test for toy TPS with AIMMD using toy_tps_aimmd.py script.

This test verifies that the toy_tps_aimmd.py script can be executed successfully.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None, timeout=300):
    """Run a command and return success/failure."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_toy_tps_imports():
    """Test toy_tps_aimmd.py imports."""
    print("\n" + "=" * 70)
    print("  Testing toy_tps_aimmd.py Imports")
    print("=" * 70 + "\n")

    script_path = Path(__file__).parent.parent / "toy_tps_aimmd.py"

    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False

    # Run with --help to check if imports work
    cmd = [sys.executable, str(script_path), "--help"]
    success, stdout, stderr = run_command(cmd)

    if success:
        print("✓ toy_tps_aimmd.py imports successful")
        return True
    else:
        print("✗ toy_tps_aimmd.py import failed")
        if stderr:
            print(f"Error: {stderr}")
        return False


def test_toy_tis_imports():
    """Test toy_tis_aimmd.py imports."""
    print("\n" + "=" * 70)
    print("  Testing toy_tis_aimmd.py Imports")
    print("=" * 70 + "\n")

    script_path = Path(__file__).parent.parent / "toy_tis_aimmd.py"

    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False

    # Run with --help to check if imports work
    cmd = [sys.executable, str(script_path), "--help"]
    success, stdout, stderr = run_command(cmd)

    if success:
        print("✓ toy_tis_aimmd.py imports successful")
        return True
    else:
        print("✗ toy_tis_aimmd.py import failed")
        if stderr:
            print(f"Error: {stderr}")
        return False


def main():
    """Run integration tests."""
    print("\n" + "=" * 70)
    print("  AIMMD-TIS Toy Examples Integration Tests")
    print("=" * 70)

    tests = [
        ("toy_tps_aimmd.py imports", test_toy_tps_imports),
        ("toy_tis_aimmd.py imports", test_toy_tis_imports),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("  Integration Test Summary")
    print("=" * 70 + "\n")

    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed

    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")

    print(f"\nTotal Passed: {passed}")
    print(f"Total Failed: {failed}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
