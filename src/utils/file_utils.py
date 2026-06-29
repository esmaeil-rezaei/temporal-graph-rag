from __future__ import annotations

import hashlib
import re
from pathlib import Path


def hash_file(file_path: Path) -> str:
    """Return SHA-256 hex digest of the file's byte content."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


def extract_version(file_path: Path) -> str | None:
    """Extract a version string (e.g. v1.2 or 2024-01-01) from the file name."""
    name = file_path.stem
    version_match = re.search(r"v(\d+(?:\.\d+)+)", name, re.IGNORECASE)
    if version_match:
        return version_match.group(0)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if date_match:
        return date_match.group(1)
    return None
