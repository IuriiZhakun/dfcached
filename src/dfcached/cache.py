from __future__ import annotations
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional, Union, Iterable
import os, time, json, pickle, hashlib, base64
import pandas as pd

# ---------- helpers ----------

def _b64(x: bytes) -> str: return base64.b64encode(x).decode("ascii")
def _b64d(s: str) -> bytes: return base64.b64decode(s.encode("ascii"))
def _is_df(x: Any) -> bool: return isinstance(x, pd.DataFrame)


def _key(
    func_name: str,
    args: tuple,
    kwargs: dict,
    version: Optional[str],
    exclude: Optional[Iterable[str]],
    canonicalize_kwargs: bool,
) -> str:
    # 1) filter kwargs
    if exclude:
        kwargs = {k: v for k, v in kwargs.items() if k not in exclude}

    # 2) canonicalize kwargs order if requested (kwargs keys are always str)
    if canonicalize_kwargs:
        kw_items = tuple(sorted(kwargs.items(), key=lambda kv: kv[0]))
        payload = (args, kw_items)
    else:
        payload = (args, kwargs)

    # 3) hash
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


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

class _DirLock:
    def __init__(self, base: Path, timeout: float = 10.0, sleep: float = 0.02):
        self.lock = base.with_suffix(base.suffix + ".lock")
        self.timeout, self.sleep, self.fd = timeout, sleep, None
    def __enter__(self):
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode()); return self
            except FileExistsError:
                if time.time() - start > self.timeout: raise TimeoutError(f"lock timeout: {self.lock}")
                time.sleep(self.sleep)
    def __exit__(self, *exc):
        try:
            if self.fd is not None: os.close(self.fd)
            if self.lock.exists(): os.unlink(self.lock)
        finally:
            self.fd = None

# ---------- I/O for top-level containers ----------

def _save_df(base: Path, label: str, df: pd.DataFrame) -> dict:
    try:
        p = base / f"{label}.parquet"
        df.to_parquet(p)  # requires pyarrow/fastparquet; else fallback
        return {"kind": "parquet", "file": p.name}
    except Exception:
        p = base / f"{label}.pkl"
        with open(p, "wb") as f: pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        return {"kind": "pickle", "file": p.name}

def _load_leaf(base: Path, entry: dict) -> Any:
    if entry["kind"] == "parquet":
        return pd.read_parquet(base / entry["file"])
    with open(base / entry["file"], "rb") as f:
        return pickle.load(f)

def _save_result(base: Path, result: Any) -> dict:
    if _is_df(result):
        e = _save_df(base, "0", result)
        return {"container": "df", "items": [e]}

    if isinstance(result, tuple):
        out = []
        for i, v in enumerate(result):
            if _is_df(v): out.append(_save_df(base, str(i), v))
            else:
                p = base / f"{i}.pkl"
                with open(p, "wb") as f: pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
                out.append({"kind": "pickle", "file": p.name})
        return {"container": "tuple", "items": out}

    if isinstance(result, list):
        out = []
        for i, v in enumerate(result):
            if _is_df(v): out.append(_save_df(base, str(i), v))
            else:
                p = base / f"{i}.pkl"
                with open(p, "wb") as f: pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
                out.append({"kind": "pickle", "file": p.name})
        return {"container": "list", "items": out}

    if isinstance(result, dict):
        out = []
        for i, (k, v) in enumerate(result.items()):
            key_b64 = base64.b64encode(pickle.dumps(k, protocol=5)).decode("ascii")
            if _is_df(v): entry = _save_df(base, str(i), v)
            else:
                p = base / f"{i}.pkl"
                with open(p, "wb") as f: pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
                entry = {"kind": "pickle", "file": p.name}
            out.append({"key_b64": key_b64, "entry": entry})
        return {"container": "dict", "items": out}

    # other single object
    p = base / "0.pkl"
    with open(p, "wb") as f: pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"container": "other", "items": [{"kind": "pickle", "file": p.name}]}

def _load_result(base: Path, meta: dict) -> Any:
    c = meta["container"]
    if c == "df":    return _load_leaf(base, meta["items"][0])
    if c == "tuple": return tuple(_load_leaf(base, e) for e in meta["items"])
    if c == "list":  return [_load_leaf(base, e) for e in meta["items"]]
    if c == "dict":
        d = {}
        for item in meta["items"]:
            k = pickle.loads(base64.b64decode(item["key_b64"].encode("ascii")))
            d[k] = _load_leaf(base, item["entry"])
        return d
    if c == "other": return _load_leaf(base, meta["items"][0])
    raise ValueError("unknown container")

# ---------- public decorator ----------

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
) -> Callable:
    """
    Minimal on-disk cache for top-level containers.
    DataFrame→Parquet (fallback→pickle); others→pickle.
    """
    cache_root = Path(cache_dir)

    def decorator(func: Callable) -> Callable:
        qual = getattr(func, "__qualname__", func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (
                 key_fn(args, kwargs)
                 if key_fn
                 else _key(qual, args, kwargs, version, exclude_from_key, canonicalize_kwargs)
             )
            base = cache_root / qual / key
            manifest = base / "manifest.json"

            if not refresh and manifest.exists():
                try:
                    meta = json.loads(manifest.read_text())
                    return _load_result(base, meta)
                except Exception:
                    pass

            base.mkdir(parents=True, exist_ok=True)
            with _DirLock(base, timeout=lock_timeout, sleep=lock_sleep):
                if not refresh and manifest.exists():  # double-check after lock
                    try:
                        meta = json.loads(manifest.read_text())
                        return _load_result(base, meta)
                    except Exception:
                        pass
                result = func(*args, **kwargs)
                meta = _save_result(base, result)
                _atomic_write_text(manifest, json.dumps(meta, indent=2))
                return result
        return wrapper
    return decorator

