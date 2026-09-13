"""Atomic local-file publication shared by experiment artifact services."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def publish_new_file(temporary_path: Path, destination: Path) -> None:
    """Atomically link a complete file into a previously unused destination."""
    try:
        os.link(temporary_path, destination)
    except FileExistsError as error:
        raise FileExistsError(f"Artifact already exists: {destination}") from error
    except OSError as error:
        if destination.exists():
            raise FileExistsError(f"Artifact already exists: {destination}") from error
        raise OSError(
            f"Could not atomically publish {destination}; the destination "
            "filesystem must support same-directory hard links."
        ) from error


def write_text_exclusive(
    destination: Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write complete text and atomically publish it without overwriting."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(content, encoding=encoding)
        publish_new_file(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
