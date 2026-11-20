"""
Smoke tests for the toy AIMMD-TPS -> AIMMD-TIS example scripts.

Run from ``examples/toy_examples``:
    pytest tests
"""

import subprocess
import sys
from pathlib import Path

import pytest

TOY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOY_DIR))


def test_package_imports_without_ops_setup():
    """aimmdTIS core must stay importable without ops-setup installed."""
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'ops_setup':\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from aimmdTIS import AIMMDSetup, AIMMD_TIS\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ["toy_tps_aimmd.py", "toy_tis_aimmd.py"])
def test_script_help(script):
    result = subprocess.run(
        [sys.executable, str(TOY_DIR / script), "--help"],
        capture_output=True, text=True, cwd=TOY_DIR,
    )
    assert result.returncode == 0, result.stderr


def test_tps_then_tis(tmp_path):
    """Short TPS run followed by TIS on one interface, using the TPS output."""
    from toy_tps_aimmd import run_toy_tps_aimmd
    from toy_tis_aimmd import run_toy_tis_aimmd

    aimmd_store, ops_store = run_toy_tps_aimmd(n_steps=3, output_path=tmp_path)
    assert aimmd_store.exists() and ops_store.exists()

    summaries = run_toy_tis_aimmd(tmp_path, interface_values=[-1.0], n_steps=2)
    assert len(summaries) == 1
    assert Path(summaries[0]["storage_path"]).exists()
    assert (tmp_path / "aimmd_tis" / "tis_descriptors_forward.meta.json").exists()
