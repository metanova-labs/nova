from __future__ import annotations

from typing import Optional


def _validate_component(value: Optional[str], label: str) -> str:
    if value is None:
        raise ValueError(f"Missing GitHub {label}. Please set the corresponding GITHUB_REPO_* variable.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"GitHub {label} cannot be empty.")
    if normalized in {".", ".."}:
        raise ValueError(f"GitHub {label} cannot be '{normalized}'.")
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"GitHub {label} must not contain path separators.")

    return normalized


def normalize_github_repo_path(repo_path: Optional[str]) -> str:
    if repo_path is None:
        return ""

    normalized = repo_path.strip()
    if not normalized:
        return ""
    if normalized.startswith(("/", "\\")):
        raise ValueError("GitHub repo path must be relative and must not start with a slash.")

    segments = normalized.split("/")
    if any(not segment or segment in {".", ".."} or "\\" in segment for segment in segments):
        raise ValueError("GitHub repo path must not contain empty, '.', '..', or backslash segments.")

    return "/".join(segments)


def build_github_path(
    owner: Optional[str], repo: Optional[str], branch: Optional[str], repo_path: Optional[str]
) -> str:
    github_path = "/".join(
        [
            _validate_component(owner, "owner"),
            _validate_component(repo, "repository name"),
            _validate_component(branch, "branch"),
        ]
    )
    normalized_repo_path = normalize_github_repo_path(repo_path)
    if normalized_repo_path:
        github_path = f"{github_path}/{normalized_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path
