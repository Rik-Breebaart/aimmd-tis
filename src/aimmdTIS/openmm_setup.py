"""
OpenMM setup wrappers for AIMMD-TIS sampling.

This module provides thin wrappers around ops-setup OpenMM host-guest
setups to avoid duplicated setup logic.
"""

from pathlib import Path
from typing import Callable, Optional

from ops_setup.systems.examples.host_guest import (
    HostGuestTPSSetup as OpsHostGuestTPSSetup,
    HostGuestTISSetup as OpsHostGuestTISSetup,
)


class TPS_setup(OpsHostGuestTPSSetup):
    """
    Host-guest TPS setup wrapper using ops-setup.

    Parameters
    ----------
    config_path : str or Path
        Path to TPS configuration file.
    resource_directory : str or Path, optional
        Directory with system resource files.
    print_config : bool, optional
        Whether to print config.
    cv_function : callable
        CV function for defining states.
    """

    def __init__(self, config_path: Path, resource_directory: Path = "",
                 print_config: bool = True, cv_function: Optional[Callable] = None):
        if cv_function is None:
            raise ValueError("cv_function is required for host-guest TPS setup")
        super().__init__(config_path, cv_function, resource_directory, print_config)


class TIS_setup(OpsHostGuestTISSetup):
    """
    Host-guest TIS setup wrapper using ops-setup.

    Parameters
    ----------
    tps_config_path : str or Path
        Path to TPS configuration file.
    tis_config_path : str or Path
        Path to TIS configuration file.
    resource_directory : str or Path, optional
        Directory with system resource files.
    print_config : bool, optional
        Whether to print config.
    cv_function : callable
        CV function for defining states.
    """

    def __init__(self, tps_config_path: Path, tis_config_path: Path,
                 resource_directory: Path = "", print_config: bool = True,
                 cv_function: Optional[Callable] = None):
        if cv_function is None:
            raise ValueError("cv_function is required for host-guest TIS setup")
        super().__init__(tps_config_path, tis_config_path, cv_function,
                         resource_directory, print_config)
