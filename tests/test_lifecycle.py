"""Unit tests for graceful drain + readiness HTTP thread (no validator / bittensor imports)."""

from __future__ import annotations

import asyncio
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from neurons.validator import lifecycle


@pytest.fixture(autouse=True)
def reset_lifecycle_state():
    lifecycle.reset_for_testing()
    yield
    lifecycle.reset_for_testing()


def _status_for_path(host: str, port: int, path: str) -> int:
    try:
        with urlopen(f"http://{host}:{port}{path}", timeout=3) as resp:
            return int(resp.status)
    except HTTPError as e:
        return int(e.code)


def test_healthz_returns_200_when_server_running():
    port = lifecycle.start_http_server(port=0, host="127.0.0.1")
    assert _status_for_path("127.0.0.1", port, "/healthz") == 200


def test_ready_returns_200_when_idle():
    port = lifecycle.start_http_server(port=0, host="127.0.0.1")
    assert _status_for_path("127.0.0.1", port, "/ready_to_update") == 200


def test_unknown_path_returns_404():
    port = lifecycle.start_http_server(port=0, host="127.0.0.1")
    assert _status_for_path("127.0.0.1", port, "/not-a-valid-lifecycle-endpoint") == 404


def test_ready_returns_423_while_epoch_scope_active():
    port = lifecycle.start_http_server(port=0, host="127.0.0.1")

    async def probe():
        async with lifecycle.epoch_busy_scope():
            return await asyncio.to_thread(_status_for_path, "127.0.0.1", port, "/ready_to_update")

    assert asyncio.run(probe()) == 423


def test_should_exit_predicate_respects_busy_flag():
    assert lifecycle.should_exit_now() is False
    lifecycle._drain_requested.set()  # noqa: SLF001 — sync with production threading.Event
    assert lifecycle.should_exit_now() is True
    lifecycle._epoch_busy.set()  # noqa: SLF001
    assert lifecycle.should_exit_now() is False
    lifecycle._epoch_busy.clear()  # noqa: SLF001
    assert lifecycle.should_exit_now() is True
