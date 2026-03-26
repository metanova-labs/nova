from __future__ import annotations

import os
import subprocess


def download_model_file(destination: str, url: str, timeout: int = 300) -> None:
    directory = os.path.dirname(destination)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        subprocess.run(
            ["wget", "-O", destination, url],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("wget is required to download model weights.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out downloading model weights from {url}.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        error_message = stderr or f"exit code {exc.returncode}"
        raise RuntimeError(f"Failed to download model weights from {url}: {error_message}") from exc
