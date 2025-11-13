# -----------------------------------------------------------------------------
# src/dfcached/_integrity.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import hashlib

__all__ = ["sha256_file"]

def sha256_file(path: Path, chunk: int = 1 << 20) -> Tuple[str, int]:
    """Return (sha256_hex, size_bytes) for file at path."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            size += len(b)
    return h.hexdigest(), size


