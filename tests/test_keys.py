import json
import time
from pathlib import Path
import pandas as pd
import pytest

from dfcached import persist_cache

def _count_key_dirs(cache_root: Path, func_qualname: str) -> int:
    func_dir = cache_root / func_qualname
    return len([p for p in func_dir.glob("*") if p.is_dir()])

def test_version_segregates_keys(tmp_path, monkeypatch):
    """Same function body + qualname; different 'version' should create different key dirs."""
    monkeypatch.chdir(tmp_path)
    cache = Path(".dfcached_cache")

    def core(n: int):
        df = pd.DataFrame({"x": range(n)})
        return df

    f_v1 = persist_cache(cache_dir=cache, version="v1")(core)
    f_v2 = persist_cache(cache_dir=cache, version="v2")(core)

    # Calls
    f_v1(3)
    f_v2(3)

    # Qualname preserved by @wraps -> both write under same func dir
    qual = getattr(core, "__qualname__", core.__name__)
    assert _count_key_dirs(cache, qual) == 2

def test_kwargs_order_insensitive(tmp_path, monkeypatch):
    """Reordering kwargs hits the same cache key when canonicalization is enabled (default)."""
    monkeypatch.chdir(tmp_path)
    cache = Path(".dfcached_cache")

    @persist_cache(cache_dir=cache, version="v1")
    def f(a=0, b=0):
        # side effect only when computing
        return {"a": a, "b": b}

    # Cold
    f(a=1, b=2)
    # Should be a hot hit even with swapped order (desired), but today creates a new key
    f(b=2, a=1)

    qual = getattr(f, "__qualname__", f.__name__)
    assert _count_key_dirs(cache, qual) == 1

def test_custom_key_fn_collapses_entries(tmp_path, monkeypatch):
    """A custom key_fn can force different inputs to share a single cache entry."""
    monkeypatch.chdir(tmp_path)
    cache = Path(".dfcached_cache")

    def key_fn(args, kwargs):
        # ignore inputs entirely
        return "constant"

    @persist_cache(cache_dir=cache, version="v1", key_fn=key_fn)
    def g(x, y=0):
        return {"sum": x + y}

    g(1, y=2)    # cold
    g(100, y=-97)  # should be hot (same key)
    qual = getattr(g, "__qualname__", g.__name__)
    assert _count_key_dirs(cache, qual) == 1

def test_non_picklable_arg_fallback_repr_same_object_hits_cache(tmp_path, monkeypatch):
    """
    When args aren't picklable (e.g., lambda), _key falls back to repr().
    Using the *same object* twice should produce the same key.
    """
    monkeypatch.chdir(tmp_path)
    cache = Path(".dfcached_cache")

    calls = []

    @persist_cache(cache_dir=cache, version="v1")
    def h(fn):
        calls.append(time.time())  # record compute
        return "ok"

    lam = lambda z: z  # same object reused
    h(lam)             # cold
    h(lam)             # hot
    assert len(calls) == 1  # computed only once

def test_kwargs_exclusion_still_holds(tmp_path, monkeypatch):
    """Sanity: exclude_from_key drops volatile kwargs."""
    monkeypatch.chdir(tmp_path)
    cache = Path(".dfcached_cache")

    @persist_cache(cache_dir=cache, version="v1", exclude_from_key=("nonce",))
    def k(x, nonce=None):
        return {"x": x}

    a = k(5, nonce=123)     # cold
    b = k(5, nonce=999)     # hot
    assert a == b

def _count_key_dirs(cache_root: Path, func_qualname: str) -> int:
    func_dir = cache_root / func_qualname
    return len([p for p in func_dir.glob("*") if p.is_dir()])

def test_kwargs_order_insensitive_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = Path(".dfcached_cache")

    @persist_cache(cache_dir=cache, version="v1")  # default canonicalize_kwargs=True
    def f(a=0, b=0): return {"a": a, "b": b}

    f(a=1, b=2)       # cold
    f(b=2, a=1)       # hot (same key)
    qual = getattr(f, "__qualname__", f.__name__)
    assert _count_key_dirs(cache, qual) == 1

def test_kwargs_order_sensitive_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = Path(".dfcached_cache")

    @persist_cache(cache_dir=cache, version="v1", canonicalize_kwargs=False)
    def g(a=0, b=0): return {"a": a, "b": b}

    g(a=1, b=2)       # cold
    g(b=2, a=1)       # treated as different key
    qual = getattr(g, "__qualname__", g.__name__)
    assert _count_key_dirs(cache, qual) == 2
