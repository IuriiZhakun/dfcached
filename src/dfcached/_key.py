# src/dfcached/_key.py
from __future__ import annotations
from typing import Optional, Iterable, Tuple, Any, Dict
import pickle
import hashlib
import base64

__all__ = ["make_key", "b64", "b64d"]

def b64(x: bytes) -> str:
    return base64.b64encode(x).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def make_key(
    func_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    version: Optional[str],
    exclude: Optional[Iterable[str]],
    canonicalize_kwargs: bool,
) -> str:
    """Stable cache key: sha256(qualname | optional version | payload).

    Payload is (args, kwargs) or (args, sorted(kwargs.items())) when canonicalize_kwargs.
    Excludes listed kwargs from the key if provided.
    Returns a 64-char hex string (256-bit digest).
    """
    if exclude:
        kwargs = {k: v for k, v in kwargs.items() if k not in exclude}

    if canonicalize_kwargs:
        kw_items = tuple(sorted(kwargs.items(), key=lambda kv: kv[0]))  # kwargs keys are str
        payload = (args, kw_items)
    else:
        payload = (args, kwargs)

    m = hashlib.sha256()
    m.update(func_name.encode("utf-8"))
    if version:
        m.update(b"|ver:" + version.encode("utf-8"))
    try:
        buf = pickle.dumps(payload, protocol=5)
    except Exception:
        buf = repr(payload).encode("utf-8")
    m.update(buf)
    return m.hexdigest()
