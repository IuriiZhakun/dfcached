# -----------------------------------------------------------------------------
# src/dfcached/decorator.py
from __future__ import annotations
from pathlib import Path
from functools import wraps
from typing import Callable, Optional, Union, Iterable
import json

from ._key import make_key
from ._io import atomic_write_text, save_result, load_result
from ._lock import DirLock

__all__ = ["persist_cache"]


def persist_cache(
    cache_dir: Union[str, Path] = ".dfcached_cache",
    *,
    key_fn: Callable[[tuple, dict], str] | None = None,
    version: Optional[str] = None,
    refresh: bool = False,
    exclude_from_key: Optional[Iterable[str]] = None,
    lock_timeout: float = 10.0,
    lock_sleep: float = 0.02,
    canonicalize_kwargs: bool = True,
    write_checksums: bool = True,
    verify_checksums: bool = True,
    strict_integrity: bool = True,
) -> Callable:
    """Minimal on-disk cache for top-level containers.

    DataFrame→Parquet (fallback→pickle); others→pickle.
    Dict keys preserved via pickled+base64.
    Checksums (sha256,size) are written & verified by default; strict by default.
    """
    cache_root = Path(cache_dir)

    def decorator(func: Callable) -> Callable:
        qual = getattr(func, "__qualname__", func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (
                key_fn(args, kwargs)
                if key_fn
                else make_key(qual, args, kwargs, version, exclude_from_key, canonicalize_kwargs)
            )
            base = cache_root / qual / key
            manifest = base / "manifest.json"

            if not refresh and manifest.exists():
                try:
                    meta = json.loads(manifest.read_text())
                    return load_result(base, meta, verify_checksums=verify_checksums, strict_integrity=strict_integrity)
                except Exception:
                    if strict_integrity:
                        raise
                    # else fall through to recompute

            base.mkdir(parents=True, exist_ok=True)
            with DirLock(base, timeout=lock_timeout, sleep=lock_sleep):
                if not refresh and manifest.exists():  # double-check under lock
                    try:
                        meta = json.loads(manifest.read_text())
                        return load_result(base, meta, verify_checksums=verify_checksums, strict_integrity=strict_integrity)
                    except Exception:
                        if strict_integrity:
                            raise
                        # else recompute below
                result = func(*args, **kwargs)
                meta = save_result(base, result, write_checksums=write_checksums)
                atomic_write_text(manifest, json.dumps(meta, indent=2))
                return result
        return wrapper

    return decorator


