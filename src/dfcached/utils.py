# src/dfcached/utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Union
import json

def read_manifest(cache_root: Union[str, Path]) -> tuple[Path, dict]:
    """
    Return (manifest_path, manifest_json) for the first manifest found under cache_root.
    """
    root = Path(cache_root)
    m = next(root.rglob("manifest.json"))
    return m, json.loads(m.read_text())

def iter_leaf_entries(meta: dict) -> Iterator[dict]:
    """
    Yield all leaf entries (dicts with `kind`/`file`[/`sha256`/`size`]) from a manifest.
    Works for containers: 'df', 'tuple', 'list', 'dict', 'other'.
    """
    c = meta["container"]
    if c in ("df", "other"):
        yield meta["items"][0]
    elif c in ("tuple", "list"):
        yield from meta["items"]
    elif c == "dict":
        for item in meta["items"]:
            yield item["entry"]
    else:
        raise ValueError(f"unknown container: {c}")

def count_key_dirs(cache_root: Union[str, Path], func_qualname: str) -> int:
    """
    Count key directories for a given function qualname under the cache root.
    """
    root = Path(cache_root) / func_qualname
    return sum(1 for p in root.glob("*") if p.is_dir())

