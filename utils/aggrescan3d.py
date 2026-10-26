"""Score a TNP heavy-atom model with the pinned Aggrescan3D tools.

The binaries are installed by install_deps.sh under
NOVA-nanobody-filter/.tools/bin. Scoring uses the filter's in-process wrapper
and does not start its HTTP server.
"""

import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(NOVA_DIR, "NOVA-nanobody-filter"))

MAX_AVERAGE = Decimal("-0.68002724866200")
MAX_PEAK = Decimal("1.58023066762925")


def tool_paths() -> tuple[Path, Path]:
    """Absolute paths of the installed Aggrescan3D and FreeSASA executables."""
    bindir = Path(NOVA_DIR) / "NOVA-nanobody-filter" / ".tools" / "bin"
    return bindir / "aggrescan3d-rs", bindir / "freesasa"


def developability_config():
    """Filter config that scores the TNP model with the documented whole-VHH caps."""
    from metanano.config import Aggrescan3DConfig, Config

    aggrescan, freesasa = tool_paths()
    config = Config()
    config.developability.aggrescan3d = Aggrescan3DConfig(
        enabled=True,
        executable=str(aggrescan),
        freesasa_executable=str(freesasa),
        max_average=MAX_AVERAGE,
        max_peak=MAX_PEAK,
    )
    return config


def score_structure(sequence: str, pdb_path: Path) -> Dict[str, Any]:
    """Score an existing heavy-atom PDB. This does not run TNP or a server."""
    from metanano.filters.aggrescan3d import Aggrescan3DFilter
    from metanano.utils.aggrescan3d_wrapper import Aggrescan3DWrapper

    config = developability_config().developability.aggrescan3d
    profile = Aggrescan3DWrapper(config.executable, config.freesasa_executable).profile(
        sequence, Path(pdb_path),
    )
    return Aggrescan3DFilter(config).analyze(sequence, profile).details


def aggrescan_rejection(result: Dict[str, Any]) -> Optional[str]:
    """Reason to drop a UID, or None when Aggrescan3D passed.

    A TNP compute error has no model, so it is left to the TNP check. A missing
    block after a completed TNP profile, a scoring error, or a failed cap rejects.
    """
    if result.get("error"):
        return None
    aggrescan = result.get("aggrescan3d")
    if not isinstance(aggrescan, dict):
        return "Aggrescan3D result missing"
    if aggrescan.get("status") == "ok" and aggrescan.get("passed") is True:
        return None
    return aggrescan.get("reason") or aggrescan.get("error") or "Aggrescan3D check failed"
