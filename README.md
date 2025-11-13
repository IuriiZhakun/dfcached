# dfcached

[![PyPI](https://img.shields.io/pypi/v/dfcached.svg)](https://pypi.org/project/dfcached/)
[![Python Versions](https://img.shields.io/pypi/pyversions/dfcached.svg)](https://pypi.org/project/dfcached/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/IuriiZhakun/dfcached/actions/workflows/workflow.yml/badge.svg)](https://github.com/IuriiZhakun/dfcached/actions/workflows/workflow.yml)

Disk‑backed function result cache for pandas‑heavy workflows.

* **Zero‑config**: `@persist_cache()` works out of the box.
* **Smart I/O**: DataFrames → Parquet (fallback to pickle); other objects → pickle.
* **Integrity by default**: per‑file `sha256` + size; strict verification on read.
* **Stable keys**: kwargs order canonicalized; optional exclusions and custom key_fn.

---

## Install

```bash
pip install dfcached
# Optional: for Parquet I/O (recommended)
pip install pyarrow  # or fastparquet
```

Python 3.11–3.12 supported.

---

## Quick start

```python
from dfcached import persist_cache
import pandas as pd

@persist_cache()  # strict_integrity=True, write/verify_checksums=True by default
def build(n: int) -> pd.DataFrame:
    return pd.DataFrame({"x": range(n)})

cold = build(10)
hot  = build(10)  # loaded from cache
assert cold.equals(hot)
```

### Mixed containers

`dfcached` understands tuples, lists and dicts. Any element that is a `pd.DataFrame` is stored as Parquet (fallback to pickle if Parquet fails); everything else is pickled.

```python
@persist_cache()
def compute(n: int):
    df = pd.DataFrame({"x": range(n)})
    return (
        df,
        ["meta", 1],
        {"df": df, "sum": df["x"].sum()},
    )

_ = compute(5)   # writes cache
_ = compute(5)   # hot hit
```

---

## Parameters

```python
@persist_cache(
  cache_dir: str | Path = ".dfcached_cache",
  *,
  key_fn: Callable[[tuple, dict], str] | None = None,
  version: str | None = None,
  refresh: bool = False,
  exclude_from_key: Iterable[str] | None = None,
  lock_timeout: float = 10.0,
  lock_sleep: float = 0.02,
  canonicalize_kwargs: bool = True,
  write_checksums: bool = True,
  verify_checksums: bool = True,
  strict_integrity: bool = True,
)
```

| Parameter                     |           Default | Meaning                                                                         |
| ----------------------------- | ----------------: | ------------------------------------------------------------------------------- |
| `cache_dir`                   | `.dfcached_cache` | Root directory for on‑disk cache.                                               |
| `version`                     |            `None` | Optional version namespace. Bump to invalidate old keys.                        |
| `refresh`                     |           `False` | Force recompute even if a cache hit exists.                                     |
| `exclude_from_key`            |            `None` | Iterable of kwarg names excluded from the key (e.g. `verbose`).                 |
| `canonicalize_kwargs`         |            `True` | Make kwargs order irrelevant for the key.                                       |
| `key_fn`                      |            `None` | Custom key generator taking `(args, kwargs)` and returning a hex string.        |
| `write_checksums`             |            `True` | Write `sha256` and file size per leaf.                                          |
| `verify_checksums`            |            `True` | Verify `sha256`/size on load.                                                   |
| `strict_integrity`            |            `True` | On mismatch, **raise** with a remediation message (don’t read corrupted files). |
| `lock_timeout` / `lock_sleep` |   `10.0` / `0.02` | Directory lock to prevent duplicate work under concurrency.                     |

---

## Integrity & corruption handling

By default, each cached file stores `sha256` and size. On load, they’re verified — if a file was modified or corrupted, you get a clear error pointing to the exact path and the manifest to edit/remove.

To bypass temporarily and recompute on mismatch:

```python
@persist_cache(strict_integrity=False)
```

> Tip: deleting the single key directory is usually best (see layout below).

---

## Cache layout

```
.dfcached_cache/
  <func qualname>/
    <key hex>/
      manifest.json             # container + leaf entries (kind, file, sha256, size)
      0.parquet|0.pkl           # DataFrame or non‑DF element
      1.pkl
      ...
```

---

## Utilities

```python
from dfcached.utils import clear, read_manifest, iter_leaf_entries, count_key_dirs

# Remove all cache
clear(".dfcached_cache")

# Remove all keys for a function
clear(".dfcached_cache", func_qualname="my_mod.my_func")

# Inspect a manifest
manifest, meta = read_manifest(".dfcached_cache")
for leaf in iter_leaf_entries(meta):
    print(leaf["kind"], leaf["file"], leaf.get("sha256"))
```

---

## Key semantics

* Keys are based on a pickle of `(args, kwargs)` plus `version` and the function qualname.
* With `canonicalize_kwargs=True`, kwargs order is ignored.
* Use `exclude_from_key` to omit non‑semantic flags (e.g., logging, progress).
* For special cases (e.g., large non‑picklable args), supply a custom `key_fn`.

> Note: pickled argument bytes can differ across Python versions or object implementations. If you need cross‑env key stability, provide a custom `key_fn`.

---

## Concurrency

A lightweight directory lock avoids duplicate work when the same key is computed concurrently. Tune with `lock_timeout` and `lock_sleep`.

---

## Development

```bash
poetry install --with dev
poetry run ruff check src tests
poetry run mypy src tests
poetry run pytest -q
```

Releasing is automated via GitHub Actions (Trusted Publishing). RC tags (`vX.Y.ZrcN`) publish to TestPyPI; final tags (`vX.Y.Z`) publish to PyPI.

---

## License

MIT © Iurii Zhakun

