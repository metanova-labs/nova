import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeLogger:
    def __init__(self) -> None:
        self.info_messages = []
        self.warning_messages = []

    def info(self, message, *args, **kwargs) -> None:
        self.info_messages.append(message)

    def warning(self, message, *args, **kwargs) -> None:
        self.warning_messages.append(message)

    def error(self, message, *args, **kwargs) -> None:
        pass


class RankingTests(unittest.TestCase):
    def _load_ranking_module(self):
        logger = _FakeLogger()
        fake_bt = types.SimpleNamespace(logging=logger)
        fake_utils = types.SimpleNamespace(calculate_dynamic_entropy=lambda **kwargs: 0.0)

        sys.modules.pop("neurons.validator.ranking", None)
        with patch.dict(sys.modules, {"bittensor": fake_bt, "utils": fake_utils}):
            module = importlib.import_module("neurons.validator.ranking")

        return module, logger

    def test_determine_winner_uses_supplied_epoch_length_in_logs(self) -> None:
        ranking, logger = self._load_ranking_module()

        winner = ranking.determine_winner(
            {
                1: {"boltz_score": 1.5, "block_submitted": 130},
                2: {"boltz_score": 3.0, "block_submitted": 725},
            },
            model_name="boltz",
            epoch_length=100,
        )

        self.assertEqual(winner, 2)
        self.assertIn("Epoch 7 boltz winner: UID=2, score=3.0, block=725", logger.info_messages)


if __name__ == "__main__":
    unittest.main()
