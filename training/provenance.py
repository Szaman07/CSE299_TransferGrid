"""Shared source-provenance helpers for experiment artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def capture_source_revision(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return the Git commit and whether tracked or untracked changes exist."""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {
        "git_commit": commit_result.stdout.strip(),
        "git_dirty": bool(status_result.stdout.strip()),
    }
