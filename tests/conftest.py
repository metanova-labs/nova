"""Test fixtures for the validator unit suite.

The validator's `utils` package re-exports modules that import bittensor and
torch at top level — pulling them in CI would require the full Docker stack
(slow, fragile). The helpers below load only the dependency-free submodules
directly from disk via importlib, bypassing utils/__init__.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, relative_path: str) -> ModuleType:
    """Load a Python module by file path, bypassing package __init__.

    Args:
        name: Logical module name (used for sys.modules and error messages).
        relative_path: Path under the repo root, e.g. "utils/fasta.py".
    """
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
