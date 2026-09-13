"""Fail-closed repository path normalization for governed artifacts.

Historical ledgers intentionally preserve the absolute workstation paths that
were recorded at execution time.  Derived analyses must not require that
workstation layout, so callers resolve those declarations through the current
repository root while retaining the original ledger bytes and hashes.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


class PortablePathError(ValueError):
    """Raised when a recorded path cannot be mapped safely into the repository."""


def repository_relative(path: str | Path, root: str | Path) -> str:
    """Return a stable POSIX repository-relative path."""

    root_path = Path(root).resolve()
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as error:
        raise PortablePathError(f"Path is outside the repository: {path}") from error
    return relative.as_posix()


def _parts_from_recorded(value: str, root_name: str) -> tuple[str, ...]:
    normalized = value.strip()
    if not normalized:
        raise PortablePathError("Recorded path is empty.")

    windows = PureWindowsPath(normalized)
    if windows.drive or normalized.startswith(("\\\\", "\\")):
        parts = windows.parts
        matches = [i for i, part in enumerate(parts) if part.casefold() == root_name.casefold()]
        if not matches:
            raise PortablePathError(
                f"Absolute recorded path does not contain repository root {root_name!r}: {value}"
            )
        relative = parts[matches[-1] + 1 :]
    else:
        relative = PurePosixPath(normalized.replace("\\", "/")).parts

    if not relative or any(part in {"", ".", ".."} for part in relative):
        raise PortablePathError(f"Unsafe repository-relative path: {value}")
    return tuple(relative)


def resolve_recorded_path(
    value: str | Path,
    root: str | Path,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a legacy absolute or relative declaration under ``root``.

    Windows paths are mapped only when they contain the repository directory
    name.  Relative declarations accept either slash convention.  Traversal
    outside the current repository is always rejected.
    """

    root_path = Path(root).resolve()
    parts = _parts_from_recorded(str(value), root_path.name)
    candidate = root_path.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as error:
        raise PortablePathError(f"Recorded path escapes repository root: {value}") from error
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Recorded repository artifact is missing: {candidate}")
    return candidate
