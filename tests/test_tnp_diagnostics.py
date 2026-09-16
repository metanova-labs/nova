"""Stdlib-only tests; do not import the validator/GPU application stack."""

import importlib.util
import ast
import logging
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "tnp_diagnostics_under_test", Path(__file__).parents[1] / "utils/tnp_diagnostics.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def event():
    return {
        "schema": 1, "event": "attempt_failed", "profile_id": "a" * 32,
        "sequence_sha256": "b" * 64, "attempt": 1, "max_attempts": 3,
        "returncode": 0, "elapsed_s": 5.1,
        "categories": ["cuda_launch_failure", "profile_missing"],
    }


def record(payload):
    item = logging.LogRecord("metanano.utils.tnp_wrapper", logging.WARNING,
                             __file__, 0, "TNP_DIAGNOSTIC %s", ("UNTRUSTED_MARKER",), None)
    item.tnp_diagnostic = payload
    return item


def test_bridge_serializes_validated_payload_not_raw_args():
    messages = []
    mod.TNPDiagnosticHandler(messages.append).emit(record(event()))
    assert len(messages) == 1
    assert "cuda_launch_failure" in messages[0]
    assert "UNTRUSTED_MARKER" not in messages[0]


@pytest.mark.parametrize("field,value", [
    ("categories", ["UNTRUSTED_MARKER"]), ("profile_id", "UNTRUSTED_MARKER"),
    ("sequence_sha256", "UNTRUSTED_MARKER"), ("event", "UNTRUSTED_MARKER"),
    ("elapsed_s", float("nan")), ("elapsed_s", True), ("attempt", "1"),
    ("attempt", 4), ("max_attempts", 0), ("returncode", "UNTRUSTED_MARKER"),
    ("extra_field", "UNTRUSTED_MARKER"),
    ("traceback_locations", [{"component": "UNTRUSTED_MARKER", "line": 10}]),
    ("traceback_locations", [{"component": "TNP", "line": "UNTRUSTED_MARKER"}]),
    ("traceback_locations", [{"component": "TNP", "line": 10, "source": "UNTRUSTED_MARKER"}]),
])
def test_bridge_rejects_untrusted_events(field, value):
    messages = []
    payload = event()
    payload[field] = value
    mod.TNPDiagnosticHandler(messages.append).emit(record(payload))
    assert messages == []


def test_bridge_ignores_legacy_raw_records():
    messages = []
    item = record(event())
    item.msg = "TNP failure: %s"
    mod.TNPDiagnosticHandler(messages.append).emit(item)
    assert messages == []


def test_bridge_relays_none_subscript_with_safe_traceback_location():
    messages = []
    payload = event()
    payload["categories"] = ["none_subscript"]
    payload["traceback_locations"] = [{"component": "TNP", "line": 280}]
    mod.TNPDiagnosticHandler(messages.append).emit(record(payload))
    assert len(messages) == 1
    assert "none_subscript" in messages[0]
    assert "280" in messages[0]
    assert "UNTRUSTED_MARKER" not in messages[0]


def test_sink_exception_does_not_dump_raw_record(capsys):
    def sink(_):
        raise RuntimeError("UNTRUSTED_MARKER")
    mod.TNPDiagnosticHandler(sink).emit(record(event()))
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_install_is_idempotent_and_uses_new_sink():
    logger = logging.getLogger("metanano.utils.tnp_wrapper")
    old_handlers, old_level = logger.handlers[:], logger.level
    try:
        logger.handlers = []
        logger.setLevel(logging.ERROR)
        first = mod.install_tnp_diagnostic_bridge(lambda _: None)
        messages = []
        second = mod.install_tnp_diagnostic_bridge(messages.append)
        assert first is second
        assert len(logger.handlers) == 1
        logger.warning("TNP_DIAGNOSTIC %s", "UNTRUSTED_MARKER", extra={"tnp_diagnostic": event()})
        assert len(messages) == 1
        assert "UNTRUSTED_MARKER" not in messages[0]
    finally:
        logger.handlers, logger.level = old_handlers, old_level


def test_runtime_fingerprints_executed_source_not_assumed_install(monkeypatch):
    class Wrapper:
        max_attempts = 3
    class Filter:
        _tnp = Wrapper()
    class Service:
        _filter = Filter()
    base = Path("/scientific-filter")
    paths = {
        Wrapper: base / "metanano/utils/tnp_wrapper.py",
        Service: base / "metanano/services/developability_service.py",
    }
    monkeypatch.setattr(mod.inspect, "getfile", lambda cls: str(paths[cls]))
    monkeypatch.setattr(mod, "_revision", lambda directory: "a" * 40)
    monkeypatch.setattr(mod, "_source_digest", lambda path: "b" * 64)
    runtime = mod.runtime_diagnostic("/validator", Service(), 16, 300)
    assert runtime["filter_is_expected_checkout"] is False
    assert runtime["max_attempts"] == 3
    assert runtime["async_waiter_timeout_s"] == 300
    assert runtime["cli_attempt_timeout_s"] == 300
    assert runtime["nominal_cli_retry_budget_s"] == 900
    assert runtime["timeout_budget_mismatch"] is True
    assert runtime["concurrency"] == 16


def test_actual_wrapper_events_reach_validator_sink(monkeypatch):
    import subprocess

    path = Path(__file__).parents[1] / "NOVA-nanobody-filter/metanano/utils/tnp_wrapper.py"
    wrapper_spec = importlib.util.spec_from_file_location("metanano.utils.tnp_wrapper", path)
    wrapper_module = importlib.util.module_from_spec(wrapper_spec)
    wrapper_spec.loader.exec_module(wrapper_module)
    monkeypatch.setattr(wrapper_module.shutil, "which", lambda _: "/usr/bin/TNP")
    monkeypatch.setattr(wrapper_module.subprocess, "run", lambda cmd, **_: subprocess.CompletedProcess(
        cmd, 0, "ERROR: NanoBodyBuilder2 failed to generate a model. UNTRUSTED_MARKER",
        "RuntimeError: CUDA error: unspecified launch failure",
    ))
    logger = logging.getLogger("metanano.utils.tnp_wrapper")
    old_handlers, old_level = logger.handlers[:], logger.level
    try:
        logger.handlers = []
        messages = []
        mod.install_tnp_diagnostic_bridge(messages.append)
        assert wrapper_module.TNPWrapper().profile("QVQL") is None
        import json
        events = [json.loads(message.removeprefix("TNP_DIAGNOSTIC ")) for message in messages]
        failures = [event for event in events if event["event"] == "attempt_failed"]
        assert [event["attempt"] for event in failures] == [1, 2, 3]
        assert all("cuda_launch_failure" in event["categories"] for event in failures)
        assert all("UNTRUSTED_MARKER" not in message for message in messages)
        assert events[-1]["event"] == "profile_exhausted"
    finally:
        logger.handlers, logger.level = old_handlers, old_level


@pytest.mark.parametrize("result,expected", [
    (None, True), ({"passed": False, "error": True, "error_kind": "timeout"}, True),
    ({"passed": False, "reason": "Failed to compute TNP profile."}, True),
    ({"passed": False, "flags": {"L": "red"}}, False), ({"passed": True}, False),
])
def test_compute_error_predicate_keeps_biological_rejections_separate(result, expected):
    path = Path(__file__).parents[1] / "neurons/validator/nanobody_validity.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_is_tnp_compute_error")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["_is_tnp_compute_error"](result) is expected
