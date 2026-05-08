"""Structural checks on docker-compose.yml — no Docker daemon required."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path(__file__).resolve().parent.parent / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose_document():
    assert COMPOSE_PATH.exists(), f"missing compose file at {COMPOSE_PATH}"
    return yaml.safe_load(COMPOSE_PATH.read_text())


def test_services_include_validator_and_watchtower(compose_document):
    services = compose_document.get("services") or {}
    assert {"validator", "watchtower"} <= set(services)


def test_validator_watchtower_placement_and_signals(compose_document):
    validator = compose_document["services"]["validator"]
    assert validator.get("stop_signal") == "SIGTERM"
    assert validator.get("stop_grace_period") == "30m"
    labels = validator.get("labels") or {}
    assert labels.get("com.centurylinklabs.watchtower.enable") == "true"
    assert labels.get("com.centurylinklabs.watchtower.lifecycle.pre-update") == "/app/nova/scripts/pre-update.sh"


def test_watchtower_lifecycle_hooks(compose_document):
    wt = compose_document["services"]["watchtower"]
    assert wt.get("image") == "nickfedor/watchtower:latest"
    cmd = wt.get("command") or []
    assert "--label-enable" in cmd
    assert "--enable-lifecycle-hooks" in cmd
    assert "--cleanup" in cmd
    assert "--interval" in cmd
    assert "300" in cmd
    assert "--notifications" in cmd
    assert "shoutrrr" in cmd
    env = wt.get("environment") or {}
    assert "WATCHTOWER_NOTIFICATION_URL" in env
    volumes = wt.get("volumes") or []
    assert any("docker.sock" in str(v) for v in volumes)
