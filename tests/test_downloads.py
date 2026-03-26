import subprocess
import unittest
from unittest.mock import patch

from neurons.downloads import download_model_file


class DownloadModelFileTests(unittest.TestCase):
    @patch("neurons.downloads.subprocess.run")
    def test_download_model_file_uses_checked_wget(self, mocked_run) -> None:
        download_model_file("/tmp/model.pt", "https://example.com/model.pt", timeout=12)

        mocked_run.assert_called_once_with(
            ["wget", "-O", "/tmp/model.pt", "https://example.com/model.pt"],
            check=True,
            capture_output=True,
            text=True,
            timeout=12,
        )

    @patch("neurons.downloads.subprocess.run")
    def test_download_model_file_surfaces_process_errors(self, mocked_run) -> None:
        mocked_run.side_effect = subprocess.CalledProcessError(
            4,
            ["wget"],
            stderr="network failed",
        )

        with self.assertRaisesRegex(RuntimeError, "network failed"):
            download_model_file("/tmp/model.pt", "https://example.com/model.pt")


if __name__ == "__main__":
    unittest.main()
