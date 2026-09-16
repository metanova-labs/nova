"""Credential-safe TNP diagnostics routed to the validator's existing log sink.

Do not bridge arbitrary Python logs: CLI stderr and exception strings can carry
sensitive data. Only our structured, independently validated event is relayed.
"""

import hashlib
import inspect
import json
import logging
import math
import re
import subprocess
from pathlib import Path


_EVENTS = {
    "profile_started", "profile_succeeded", "profile_exhausted",
    "attempt_started", "attempt_succeeded", "attempt_failed",
}
_CATEGORIES = {
    "executable_missing", "budget_exhausted", "cuda_launch_failure", "cuda_oom",
    "cuda_illegal_memory", "cuda_device_assert", "model_prediction_failed",
    "openmm_error", "import_error", "process_failed", "timeout",
    "process_start_failed", "profile_invalid", "profile_missing",
    "none_subscript",
}
_COMPONENTS = {
    "TNP", "tnp_cdr_assigner", "tnp_cdr3_compactness", "tnp_surface_properties",
    "tnp_pdb_utils", "immunebuilder_abodybuilder2", "immunebuilder_nanobodybuilder2",
    "immunebuilder_nbbuilder2", "immunebuilder_sequence_checks", "anarci",
}
_INTEGER_FIELDS = {"schema", "attempt", "max_attempts", "returncode", "sequence_length"}
_TIME_FIELDS = {"elapsed_s", "timeout_s", "total_timeout_s"}
_FIELDS = _INTEGER_FIELDS | _TIME_FIELDS | {
    "profile_id", "sequence_sha256", "event", "categories", "traceback_locations",
}


def safe_event(payload):
    """Rebuild allowlisted fields; never serialize an untrusted raw record."""
    if not isinstance(payload, dict) or set(payload) - _FIELDS:
        return None
    if payload.get("schema") != 1 or payload.get("event") not in _EVENTS:
        return None
    if not isinstance(payload.get("profile_id"), str) or not re.fullmatch(
        r"[0-9a-f]{32}", payload["profile_id"]
    ):
        return None
    if not isinstance(payload.get("sequence_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", payload["sequence_sha256"]
    ):
        return None
    if type(payload.get("attempt")) is not int or type(payload.get("max_attempts")) is not int:
        return None
    if not 0 <= payload["attempt"] <= payload["max_attempts"] <= 10000:
        return None
    if payload["max_attempts"] < 1:
        return None
    result = {}
    for key, value in payload.items():
        if key in _INTEGER_FIELDS:
            if type(value) is not int:
                return None
        elif key in _TIME_FIELDS:
            if value is None and key == "total_timeout_s":
                pass
            elif type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                return None
        elif key == "categories":
            if not isinstance(value, list) or any(
                not isinstance(item, str) or item not in _CATEGORIES for item in value
            ):
                return None
            value = list(value)
        elif key == "traceback_locations":
            if not isinstance(value, list) or len(value) > 16:
                return None
            safe_locations = []
            for location in value:
                if not isinstance(location, dict) or set(location) != {"component", "line"}:
                    return None
                component, line = location["component"], location["line"]
                if not isinstance(component, str) or component not in _COMPONENTS:
                    return None
                if type(line) is not int or not 1 <= line <= 9999999:
                    return None
                safe_locations.append({"component": component, "line": line})
            value = safe_locations
        result[key] = value
    return result


class TNPDiagnosticHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record):
        try:
            if record.msg != "TNP_DIAGNOSTIC %s":
                return
            event = safe_event(getattr(record, "tnp_diagnostic", None))
            if event is not None:
                self.sink("TNP_DIAGNOSTIC " + json.dumps(event, sort_keys=True))
        except Exception:
            # logging.handleError would dump the raw record/args to stderr.
            # Never expose a rejected event or a sink exception.
            return


def install_tnp_diagnostic_bridge(sink):
    logger = logging.getLogger("metanano.utils.tnp_wrapper")
    for handler in logger.handlers:
        if isinstance(handler, TNPDiagnosticHandler):
            handler.sink = sink
            return handler
    handler = TNPDiagnosticHandler(sink)
    logger.addHandler(handler)
    # A WARNING event must not disappear because a third-party logger was
    # configured at ERROR. This affects this one logger, not global verbosity.
    logger.setLevel(logging.WARNING)
    return handler


def _revision(directory):
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            timeout=2, check=False,
        )
        value = completed.stdout.strip()
        return value if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _source_digest(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def runtime_diagnostic(nova_dir, service, concurrency, total_timeout):
    """Exact checkout and executed-source fingerprints, never a config/env dump."""
    nova_dir = Path(nova_dir)
    wrapper = service._filter._tnp
    wrapper_source = Path(inspect.getfile(type(wrapper))).resolve()
    service_source = Path(inspect.getfile(type(service))).resolve()
    filter_dir = wrapper_source.parents[2]
    return {
        "schema": 1,
        "nova_revision": _revision(nova_dir),
        "filter_revision": _revision(filter_dir),
        "filter_is_expected_checkout": filter_dir == (nova_dir / "NOVA-nanobody-filter").resolve(),
        "wrapper_source_sha256": _source_digest(wrapper_source),
        "service_source_sha256": _source_digest(service_source),
        "max_attempts": int(wrapper.max_attempts),
        "concurrency": int(concurrency),
        "async_waiter_timeout_s": float(total_timeout),
        "cli_attempt_timeout_s": 300.0,
        "nominal_cli_retry_budget_s": 300.0 * wrapper.max_attempts,
        "timeout_budget_mismatch": 300.0 * wrapper.max_attempts > float(total_timeout),
    }
