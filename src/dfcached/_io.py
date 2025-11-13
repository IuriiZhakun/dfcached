from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import pickle

import pandas as pd

from ._integrity import sha256_file
from ._key import b64, b64d

__all__ = [
    "atomic_write_text",
    "save_df",
    "load_leaf",
    "save_result",
    "load_result",
]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_df(base: Path, label: str, df: pd.DataFrame, *, write_checksums: bool) -> Dict[str, Any]:
    """
    Write a DataFrame as parquet (fallback to pickle), return manifest leaf node.
    """
    p: Path
    kind: str
    try:
        p = base / f"{label}.parquet"
        df.to_parquet(p)
        kind = "parquet"
    except Exception:
        p = base / f"{label}.pkl"
        with open(p, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        kind = "pickle"

    node: Dict[str, Any] = {"kind": kind, "file": p.name}
    if write_checksums:
        sha, size = sha256_file(p)
        node["sha256"], node["size"] = sha, size
    return node


def load_leaf(
    base: Path,
    entry: Dict[str, Any],
    *,
    verify_checksums: bool = False,
    strict_integrity: bool = True,
) -> Any:
    path = base / entry["file"]
    if verify_checksums and "sha256" in entry:
        sha, size = sha256_file(path)
        expected_sha = entry["sha256"]
        expected_sz = entry.get("size", size)
        if sha != expected_sha or size != expected_sz:
            manifest_path = base / "manifest.json"
            msg = (
                "Cache integrity check failed\n"
                f"  file: {str(path)}\n"
                f"  kind: {entry.get('kind','?')}\n"
                f"  expected: sha256={expected_sha} size={expected_sz}\n"
                f"  got:      sha256={sha} size={size}\n"
                f"strict_integrity={strict_integrity}: refusing to load cached result.\n"
                "Fix one of the following:\n"
                "  1) Delete the corrupted file or its parent cache key directory and retry.\n"
                "  2) Re-run with strict_integrity=False to recompute and overwrite.\n"
                "  3) If you intentionally modified the file, update the stored checksum in:\n"
                f"       {str(manifest_path)}\n"
            )
            raise ValueError(msg)
    if entry["kind"] == "parquet":
        return pd.read_parquet(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_result(base: Path, result: Any, *, write_checksums: bool) -> Dict[str, Any]:
    """
    Persist a supported result and return the manifest meta.
    - DataFrame -> parquet (fallback pickle)
    - tuple/list/dict: DataFrame elements to parquet else pickle
    - other single object -> pickle
    """
    if isinstance(result, pd.DataFrame):
        e = save_df(base, "0", result, write_checksums=write_checksums)
        return {"container": "df", "items": [e]}

    if isinstance(result, tuple):
        items_tuple: list[Dict[str, Any]] = []
        for i, v in enumerate(result):
            if isinstance(v, pd.DataFrame):
                node = save_df(base, str(i), v, write_checksums=write_checksums)
            else:
                p = base / f"{i}.pkl"
                with open(p, "wb") as f:
                    pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
                node = {"kind": "pickle", "file": p.name}
                if write_checksums:
                    sha, size = sha256_file(p)
                    node["sha256"], node["size"] = sha, size
            items_tuple.append(node)
        return {"container": "tuple", "items": items_tuple}

    if isinstance(result, list):
        items_list: list[Dict[str, Any]] = []
        for i, v in enumerate(result):
            if isinstance(v, pd.DataFrame):
                node = save_df(base, str(i), v, write_checksums=write_checksums)
            else:
                p = base / f"{i}.pkl"
                with open(p, "wb") as f:
                    pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
                node = {"kind": "pickle", "file": p.name}
                if write_checksums:
                    sha, size = sha256_file(p)
                    node["sha256"], node["size"] = sha, size
            items_list.append(node)
        return {"container": "list", "items": items_list}

    if isinstance(result, dict):
        items_dict: list[Dict[str, Any]] = []
        for i, (k, v) in enumerate(result.items()):
            key_b64 = b64(pickle.dumps(k, protocol=5))  # preserve key type
            if isinstance(v, pd.DataFrame):
                entry_node = save_df(base, str(i), v, write_checksums=write_checksums)
            else:
                p = base / f"{i}.pkl"
                with open(p, "wb") as f:
                    pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
                entry_node = {"kind": "pickle", "file": p.name}
                if write_checksums:
                    sha, size = sha256_file(p)
                    entry_node["sha256"], entry_node["size"] = sha, size
            items_dict.append({"key_b64": key_b64, "entry": entry_node})
        return {"container": "dict", "items": items_dict}

    # other single object
    p = base / "0.pkl"
    with open(p, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    node = {"kind": "pickle", "file": p.name}
    if write_checksums:
        sha, size = sha256_file(p)
        node["sha256"], node["size"] = sha, size
    return {"container": "other", "items": [node]}


def load_result(base: Path, meta: Dict[str, Any], *, verify_checksums: bool, strict_integrity: bool) -> Any:
    c = meta["container"]
    if c == "df":
        return load_leaf(base, meta["items"][0], verify_checksums=verify_checksums, strict_integrity=strict_integrity)
    if c == "tuple":
        return tuple(
            load_leaf(base, e, verify_checksums=verify_checksums, strict_integrity=strict_integrity)
            for e in meta["items"]
        )
    if c == "list":
        return [
            load_leaf(base, e, verify_checksums=verify_checksums, strict_integrity=strict_integrity)
            for e in meta["items"]
        ]
    if c == "dict":
        d: Dict[Any, Any] = {}
        for item in meta["items"]:
            # decode key: base64 -> bytes -> original object
            k = pickle.loads(b64d(item["key_b64"]))
            d[k] = load_leaf(
                base,
                item["entry"],
                verify_checksums=verify_checksums,
                strict_integrity=strict_integrity,
            )
        return d
    if c == "other":
        return load_leaf(base, meta["items"][0], verify_checksums=verify_checksums, strict_integrity=strict_integrity)
    raise ValueError("unknown container")

