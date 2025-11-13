from __future__ import annotations
from pathlib import Path
from typing import Iterator, Union
import json
import shutil

def read_manifest(cache_root: Union[str, Path]) -> tuple[Path, dict]:
    root = Path(cache_root)
    m = next(root.rglob("manifest.json"))
    return m, json.loads(m.read_text())

def iter_leaf_entries(meta: dict) -> Iterator[dict]:
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
    root = Path(cache_root) / func_qualname
    return sum(1 for p in root.glob("*") if p.is_dir())

def clear(cache_dir: Union[str, Path],
          func_qualname: str | None = None,
          key: str | None = None) -> int:
    """
    Remove cached data.

    - clear(cache_dir): remove the entire cache directory (returns 1 if removed, else 0)
    - clear(cache_dir, func_qualname): remove all keys for that function (returns 1 if removed, else 0)
    - clear(cache_dir, func_qualname, key): remove that specific key directory (returns 1 if removed, else 0)
    """
    root = Path(cache_dir)
    if func_qualname and key:
        p = root / func_qualname / key
        if p.exists():
            shutil.rmtree(p)
            return 1
        return 0
    if func_qualname:
        p = root / func_qualname
        if p.exists():
            shutil.rmtree(p)
            return 1
        return 0
    if root.exists():
        shutil.rmtree(root)
        return 1
    return 0

