# tests/test_integrity.py
import json
import pytest
import pandas as pd
from dfcached import persist_cache
from dfcached.utils import read_manifest, iter_leaf_entries


def test_checksums_strict_raises_with_clear_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1")  # strict by default
    def make():
        return {"df": pd.DataFrame({"x": [1, 2, 3]})}

    make()  # cold
    manifest, meta = read_manifest(".dfcached_cache")
    entry = next(iter_leaf_entries(meta))
    data_file = manifest.parent / entry["file"]

    # corrupt the data file
    with open(data_file, "r+b") as f:
        f.seek(0)
        f.write(b"\x00\x00\x00CORRUPTED")

    with pytest.raises(ValueError) as ei:
        make()
    msg = str(ei.value)
    assert "Cache integrity check failed" in msg
    assert str(data_file) in msg
    assert "strict_integrity=True" in msg
    assert "Delete the corrupted file" in msg
    assert "strict_integrity=False" in msg
    assert str(manifest) in msg


def test_checksums_non_strict_autoheals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1", strict_integrity=False)
    def make():
        return {"df": pd.DataFrame({"x": [1, 2, 3]})}

    make()  # cold
    manifest, meta = read_manifest(".dfcached_cache")
    entry = next(iter_leaf_entries(meta))
    data_file = manifest.parent / entry["file"]

    with open(data_file, "r+b") as f:
        f.seek(0)
        f.write(b"\x00BAD")

    # Should recompute and NOT raise
    out = make()
    assert list(out["df"]["x"]) == [1, 2, 3]


def test_manifest_contains_checksums_for_all_tuple_leaves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1")
    def build():
        df = pd.DataFrame({"x": [0, 1]})
        return (df, ["a", 1], {"k": 7}, 42)

    build()

    manifest, meta = read_manifest(".dfcached_cache")
    assert meta["container"] == "tuple"
    for leaf in iter_leaf_entries(meta):
        assert "sha256" in leaf and "size" in leaf


def test_verify_disabled_ignores_manifest_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1",
                   write_checksums=True, verify_checksums=False)
    def f():
        return pd.DataFrame({"x": [1, 2]})

    f()

    manifest, meta = read_manifest(".dfcached_cache")
    assert meta["container"] == "df"
    leaf = next(iter_leaf_entries(meta))
    leaf["sha256"] = "0" * 64
    manifest.write_text(json.dumps(meta, indent=2))

    out = f()  # should NOT raise (verification disabled)
    assert list(out["x"]) == [1, 2]


def test_checksums_disabled_writes_no_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1",
                   write_checksums=False, verify_checksums=True)
    def g():
        return {"df": pd.DataFrame({"x": [9]})}

    g()

    _, meta = read_manifest(".dfcached_cache")
    entry = next(iter_leaf_entries(meta))
    assert "sha256" not in entry and "size" not in entry


def test_refresh_bypasses_corrupt_cache_and_rewrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1")
    def h():
        return pd.DataFrame({"x": [1, 2, 3]})

    h()

    manifest, meta = read_manifest(".dfcached_cache")
    leaf = next(iter_leaf_entries(meta))
    data_file = manifest.parent / leaf["file"]
    with open(data_file, "r+b") as f:
        f.seek(0)
        f.write(b"\x00\x00BADBADBAD")

    @persist_cache(cache_dir=".dfcached_cache", version="v1", refresh=True)
    def h_refresh():
        return pd.DataFrame({"x": [4, 5, 6]})

    out = h_refresh()
    assert list(out["x"]) == [4, 5, 6]

    _, meta2 = read_manifest(".dfcached_cache")
    leaf2 = next(iter_leaf_entries(meta2))
    assert "sha256" in leaf2 and "size" in leaf2

