# -----------------------------------------------------------------------------
# src/dfcached/_lock.py
from __future__ import annotations
from pathlib import Path
import os
import time

__all__ = ["DirLock"]

class DirLock:
    """Simple per-key directory lock using an atomic lockfile create."""

    def __init__(self, base: Path, timeout: float = 10.0, sleep: float = 0.02):
        self.lock = base.with_suffix(base.suffix + ".lock")
        self.timeout, self.sleep, self.fd = timeout, sleep, None

    def __enter__(self):
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"lock timeout: {self.lock}")
                time.sleep(self.sleep)

    def __exit__(self, *exc):
        try:
            if self.fd is not None:
                os.close(self.fd)
            if self.lock.exists():
                os.unlink(self.lock)
        finally:
            self.fd = None



