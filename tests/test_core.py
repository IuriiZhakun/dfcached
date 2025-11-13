from dfcached import persist_cache
import pandas as pd
import time
import json
from pathlib import Path

def test_hot_hit_and_tuple_list_dict_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1")
    def build(n: int):
        df = pd.DataFrame({"x": range(n), "y": [i*i for i in range(n)]})
        return (df, [1, {"k": "v"}], {"a": df, 42: ("t", 3)})

    a1 = build(5)   # cold
    a2 = build(5)   # hot
    # tuple[0] and dict["a"] are DataFrames
    assert a1[0].equals(a2[0])
    assert a1[2]["a"].equals(a2[2]["a"])
    # lists/dicts equality
    assert a1[1] == a2[1]
    assert set(a2[2].keys()) == {"a", 42}
    assert isinstance(a2[2]["a"], pd.DataFrame)
    assert a1[2][42] == a2[2][42]

def test_exclude_from_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1", exclude_from_key=("seed",))
    def f(n: int, seed: int = 0):
        return {"n": n, "fixed": 1}

    a = f(7, seed=1)       # cold
    b = f(7, seed=999)     # hot (seed excluded from key)
    assert a == b

def test_refresh_true_forces_recompute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1", refresh=True)
    def tick():
        return {"t": time.time()}

    x = tick()
    y = tick()
    assert x != y  # recomputed each time

def test_missing_manifest_recovers(tmp_path, monkeypatch):
    """If manifest is missing (e.g., interrupted write), the next call recomputes and writes a new one."""
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1")
    def g():
        return {"v": 1}

    g()  # create cache
    # locate manifest
    cache_root = Path(".dfcached_cache")
    manifests = list(cache_root.rglob("manifest.json"))
    assert len(manifests) == 1
    # delete manifest to simulate interruption
    manifests[0].unlink()
    # next call should succeed and recreate manifest
    out = g()
    assert out == {"v": 1}
    assert (cache_root).exists()
    assert list(cache_root.rglob("manifest.json")), "manifest should be recreated"

def test_dataframe_parquet_fallback_via_monkeypatch(tmp_path, monkeypatch):
    """Force DataFrame.to_parquet to fail to ensure we fall back to pickle."""
    monkeypatch.chdir(tmp_path)

    # monkeypatch pandas DF.to_parquet to simulate missing engine
    original_to_parquet = pd.DataFrame.to_parquet
    def boom(*args, **kwargs):
        raise ImportError("no parquet engine")
    pd.DataFrame.to_parquet = boom

    @persist_cache(cache_dir=".dfcached_cache", version="v1")
    def h():
        df = pd.DataFrame({"x": [1, 2, 3]})
        return {"df": df}

    try:
        h()
        # inspect manifest: kind for DF should be 'pickle'
        manifest = next(Path(".dfcached_cache").rglob("manifest.json"))
        meta = json.loads(manifest.read_text())
        items = meta["items"] if "items" in meta else meta.get("root", {}).get("items", [])
        # dict schema stores [{'key_b64':..., 'entry': {...}}]
        entry = items[0]["entry"]
        assert entry["kind"] == "pickle"
    finally:
        pd.DataFrame.to_parquet = original_to_parquet

