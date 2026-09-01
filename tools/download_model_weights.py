#!/usr/bin/env python3
"""Download NanoBodyBuilder2 and Boltz-2 weights required by the validator."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable

NOVA_DIR = Path(__file__).resolve().parent.parent
BOLTZ_SRC = NOVA_DIR / "external_tools" / "boltz" / "src"

NANOBODY_MODEL_URLS = {
    "nanobody_model_1": "https://zenodo.org/record/7258553/files/nanobody_model_1?download=1",
    "nanobody_model_2": "https://zenodo.org/record/7258553/files/nanobody_model_2?download=1",
    "nanobody_model_3": "https://zenodo.org/record/7258553/files/nanobody_model_3?download=1",
    "nanobody_model_4": "https://zenodo.org/record/7258553/files/nanobody_model_4?download=1",
}

EXPECTED_MIN_BYTES = {
    "nanobody_model_1": 50 * 1024 * 1024,    # ~59 MB
    "nanobody_model_2": 180 * 1024 * 1024,   # ~205 MB
    "nanobody_model_3": 180 * 1024 * 1024,
    "nanobody_model_4": 180 * 1024 * 1024,
}


def default_nanobody_weights_dir() -> Path:
    try:
        import ImmuneBuilder
        return Path(ImmuneBuilder.__file__).resolve().parent / "trained_model"
    except ImportError as e:
        raise RuntimeError(
            "ImmuneBuilder is not installed. Run this script with the nova venv Python."
        ) from e


def _nanobody_weights_ok(path: Path, name: str) -> bool:
    if not path.is_file() or path.stat().st_size < EXPECTED_MIN_BYTES[name]:
        return False
    try:
        import torch
        torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def download_nanobody_builder_weights(
    dest_dir: str | os.PathLike | None = None,
    *,
    models: Iterable[str] = NANOBODY_MODEL_URLS,
    force: bool = False,
) -> dict[str, Path]:
    dest = Path(dest_dir) if dest_dir is not None else default_nanobody_weights_dir()
    dest.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    for name in models:
        if name not in NANOBODY_MODEL_URLS:
            raise ValueError(f"Unknown model {name!r}. Choose from {list(NANOBODY_MODEL_URLS)}")

        path = dest / name
        if not force and _nanobody_weights_ok(path, name):
            print(f"Skipping {name} (already present and valid)")
            downloaded[name] = path
            continue

        url = NANOBODY_MODEL_URLS[name]
        fd, tmp = tempfile.mkstemp(prefix=f"{name}.", suffix=".partial", dir=dest)
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            print(f"Downloading {name} -> {path}")
            with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            if not _nanobody_weights_ok(tmp_path, name):
                raise RuntimeError(f"{name} downloaded but failed integrity check")
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        downloaded[name] = path

    return downloaded


def download_boltz2_weights(cache: str | os.PathLike | None = None) -> Path:
    """Download Boltz-2 CCD mols + conf/affinity checkpoints via boltz.main.download_boltz2."""
    if str(BOLTZ_SRC) not in sys.path:
        sys.path.insert(0, str(BOLTZ_SRC))

    from boltz.main import download_boltz2, get_cache_path

    cache_path = Path(cache).expanduser() if cache is not None else Path(get_cache_path())
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Boltz-2 weights into {cache_path}")
    download_boltz2(cache_path)
    return cache_path


def download_all_weights(
    *,
    nanobody_dir: str | os.PathLike | None = None,
    boltz_cache: str | os.PathLike | None = None,
    force_nanobody: bool = False,
) -> dict[str, Path]:
    nanobody = download_nanobody_builder_weights(nanobody_dir, force=force_nanobody)
    boltz_cache_path = download_boltz2_weights(boltz_cache)
    return {
        **nanobody,
        "boltz2_cache": boltz_cache_path,
        "boltz2_conf": boltz_cache_path / "boltz2_conf.ckpt",
        "boltz2_aff": boltz_cache_path / "boltz2_aff.ckpt",
        "boltz2_mols": boltz_cache_path / "mols",
    }


if __name__ == "__main__":
    for name, path in download_all_weights().items():
        size = path.stat().st_size if path.is_file() else "dir"
        print(f"{name}: {path} ({size})")
