"""Sanity checks on the release workflow itself.

Defends against accidental regressions to the gate (e.g. someone removing the
`needs:` chain so push happens before smoke, or dropping a required tag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release.yml"


@pytest.fixture(scope="module")
def workflow():
    assert WORKFLOW_PATH.exists(), f"missing workflow at {WORKFLOW_PATH}"
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def test_workflow_is_valid_yaml(workflow):
    assert isinstance(workflow, dict)
    assert workflow.get("name") == "release"


def test_triggers_include_push_main_and_pull_request(workflow):
    # PyYAML parses the `on:` key as a Python `True` because YAML 1.1
    # interprets bare "on" as a boolean. Accept either.
    on = workflow.get("on") or workflow.get(True)
    assert on, "missing trigger config"
    assert on.get("push", {}).get("branches") == ["main"]
    assert "pull_request" in on


def test_required_jobs_present(workflow):
    jobs = workflow.get("jobs", {})
    assert set(jobs.keys()) >= {"lint", "tests", "build-and-smoke"}


def test_build_depends_on_lint_and_tests(workflow):
    needs = workflow["jobs"]["build-and-smoke"].get("needs", [])
    assert set(needs) == {"lint", "tests"}, (
        "build-and-smoke must wait for both lint and tests"
    )


def test_push_steps_are_gated_to_main(workflow):
    """Push must only happen on push events to main, never on pull_request."""
    steps = workflow["jobs"]["build-and-smoke"]["steps"]
    push_steps = [
        s for s in steps
        if "docker/login-action" in str(s.get("uses", ""))
        or s.get("name", "").lower().startswith("push")
    ]
    assert push_steps, "expected at least one push-related step"
    for step in push_steps:
        guard = step.get("if", "")
        assert "github.event_name == 'push'" in guard, (
            f"step {step.get('name')!r} must be gated on push event"
        )
        assert "refs/heads/main" in guard, (
            f"step {step.get('name')!r} must be gated on main branch"
        )


def test_metadata_action_emits_required_tags(workflow):
    """Tags latest, main, sha-<full>, sha-<short> are all required."""
    steps = workflow["jobs"]["build-and-smoke"]["steps"]
    meta = next(s for s in steps if "docker/metadata-action" in str(s.get("uses", "")))
    raw_tags = meta["with"]["tags"]
    assert "type=raw,value=latest" in raw_tags
    assert "type=raw,value=main" in raw_tags
    assert "type=sha,prefix=sha-,format=long" in raw_tags
    assert "type=sha,prefix=sha-,format=short" in raw_tags


def test_smoke_runs_before_push(workflow):
    """The smoke import step must come before the push step in declaration
    order; if smoke ever moves after push, the gate is meaningless."""
    steps = workflow["jobs"]["build-and-smoke"]["steps"]
    smoke_idx = next(
        i for i, s in enumerate(steps)
        if s.get("name", "").startswith("Smoke — import")
    )
    push_idx = next(
        i for i, s in enumerate(steps)
        if s.get("name", "").startswith("Push")
    )
    assert smoke_idx < push_idx


def test_concurrency_cancels_in_progress(workflow):
    """Force-pushes to a PR shouldn't stack expensive builds."""
    concurrency = workflow.get("concurrency", {})
    assert concurrency.get("cancel-in-progress") is True


def test_permissions_are_minimal(workflow):
    """Top-level token must only have `contents: read` — push uses Docker Hub
    creds, not the GITHUB_TOKEN."""
    perms = workflow.get("permissions", {})
    assert perms == {"contents": "read"}
