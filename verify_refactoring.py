#!/usr/bin/env python
"""
Quick verification script to test all new refactored modules can be imported.

Run this after refactoring to verify no import errors.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all new module imports."""
    
    print("=" * 60)
    print("AIMMD-TIS Refactoring Import Verification")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import AIMMDSetup
    try:
        from aimmdTIS import AIMMDSetup
        print("✓ AIMMDSetup imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ AIMMDSetup import failed: {e}")
        tests_failed += 1
    
    # Test 2: Import TPS_setup
    try:
        from aimmdTIS import TPS_setup
        print("✓ TPS_setup imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ TPS_setup import failed: {e}")
        tests_failed += 1
    
    # Test 3: Import TIS_setup
    try:
        from aimmdTIS import TIS_setup
        print("✓ TIS_setup imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ TIS_setup import failed: {e}")
        tests_failed += 1
    
    # Test 4: Import load_initial_trajectory
    try:
        from aimmdTIS import load_initial_trajectory
        print("✓ load_initial_trajectory imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ load_initial_trajectory import failed: {e}")
        tests_failed += 1
    
    # Test 5: Verify AIMMDSetup has required methods
    try:
        from aimmdTIS import AIMMDSetup
        required_methods = [
            'setup_RCModel',
            'load_RCModel',
            'setup_selector',
            '_setup_torch_model',
            '_select_activation',
        ]
        for method in required_methods:
            if not hasattr(AIMMDSetup, method):
                raise ValueError(f"Missing method: {method}")
        print("✓ AIMMDSetup has all required methods")
        tests_passed += 1
    except Exception as e:
        print(f"✗ AIMMDSetup method check failed: {e}")
        tests_failed += 1
    
    # Test 6: Verify TPS_setup inherits from correct base
    try:
        from aimmdTIS import TPS_setup
        from ops_setup.systems.examples.host_guest import HostGuestTPSSetup
        if issubclass(TPS_setup, HostGuestTPSSetup):
            print("✓ TPS_setup correctly inherits from HostGuestTPSSetup")
            tests_passed += 1
        else:
            raise ValueError("TPS_setup does not inherit from HostGuestTPSSetup")
    except Exception as e:
        print(f"✗ TPS_setup inheritance check failed: {e}")
        tests_failed += 1
    
    # Test 7: Verify TIS_setup inherits from correct base
    try:
        from aimmdTIS import TIS_setup
        from ops_setup.systems.examples.host_guest import HostGuestTISSetup
        if issubclass(TIS_setup, HostGuestTISSetup):
            print("✓ TIS_setup correctly inherits from HostGuestTISSetup")
            tests_passed += 1
        else:
            raise ValueError("TIS_setup does not inherit from HostGuestTISSetup")
    except Exception as e:
        print(f"✗ TIS_setup inheritance check failed: {e}")
        tests_failed += 1
    
    print("=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
