"""Graceful shutdown and HTTP readiness probes for Watchtower-triggered updates.

Safe point semantics: callers must hold ``epoch_busy_scope()`` while executing
anything that must not race a container recycle (validator scoring epoch,
``set_weights``, and payout dispatch).

``GET /ready_to_update`` returns 423 while an epoch boundary is executing; HTTP
200 when idle so Watchtower's pre-update hook can exit 0 and proceed.

``GET /healthz`` answers 200 if the readiness server thread is responding.
"""

from __future__ import annotations

import asyncio
import json
import signal
import socket
import sys
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

_drain_requested = threading.Event()
_epoch_busy = threading.Event()

_server: HTTPServer | None = None
_server_thread: threading.Thread | None = None


def drain_requested() -> bool:
    return _drain_requested.is_set()


def epoch_busy_set() -> bool:
    """True between epoch boundary scoring and weight payouts."""
    return _epoch_busy.is_set()


def should_exit_now() -> bool:
    """True when draining and not inside a critical scoring section."""
    return _drain_requested.is_set() and not _epoch_busy.is_set()


@asynccontextmanager
async def epoch_busy_scope():
    _epoch_busy.set()
    try:
        yield
    finally:
        _epoch_busy.clear()


class _LifecycleHandler(BaseHTTPRequestHandler):
    def log_message(self, format_: str, *args: Any) -> None:
        return

    def _send_json(self, code: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            body = {"status": "ok", "busy": epoch_busy_set(), "drain": drain_requested()}
            self._send_json(200, body)
        elif path == "/ready_to_update":
            # Watchtower executes the pre-update script inside this container before stop.
            if epoch_busy_set():
                self._send_json(423, {"ready": False, "reason": "epoch_scoring_boundary"})
                return
            if drain_requested():
                self._send_json(200, {"ready": True, "reason": "draining_idle"})
                return
            self._send_json(200, {"ready": True, "reason": "idle_between_epochs"})
        else:
            self.send_error(404, "Unknown path")


def install_async_signal_handlers() -> None:
    """SIGINT/SIGTERM toggle drain flag. Calls must run inside a running asyncio loop (Unix).
    """
    if sys.platform == "win32":
        return

    loop = asyncio.get_running_loop()

    def drain() -> None:
        _drain_requested.set()
        print(
            "[nova-validator] SIGTERM/SIGINT received — draining; "
            "exit after current epoch boundary completes.",
            flush=True,
        )

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, drain)


def start_http_server(port: int | None = 8080, host: str = "0.0.0.0") -> int:
    """Start a daemon thread listening for readiness probes.

    Pass ``port=0`` with ``host=\"127.0.0.1\"`` for tests — returns the ephemeral port.
    """
    global _server, _server_thread
    if _server is not None:
        return int(_server.server_address[1])
    bind_port = port if port is not None else 8080
    srv = HTTPServer((host, bind_port), _LifecycleHandler)
    srv.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    assigned = int(srv.server_address[1])
    thread = threading.Thread(target=srv.serve_forever, name="lifecycle-http", daemon=True)
    thread.start()
    _server = srv
    _server_thread = thread
    print(f"[nova-validator] Readiness listening on http://{host}:{assigned}/", flush=True)
    return assigned


def shutdown_http_server() -> None:
    """Shut down readiness server (used by tests only)."""
    global _server, _server_thread
    if _server is None:
        return
    srv, _server = _server, None
    srv.shutdown()
    if _server_thread is not None:
        _server_thread.join(timeout=2.0)
    _server_thread = None


def reset_for_testing() -> None:
    """Reset internal state — tests only."""
    _drain_requested.clear()
    _epoch_busy.clear()
    shutdown_http_server()
