import json
from pathlib import Path
import pandas as pd
from dfcached import persist_cache
from dfcached.utils import clear, count_key_dirs, read_manifest

CACHE = ".dfcached_cache"

def test_clear_specific_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=CACHE, version="v1")
    def f(n: int):
        # ensure distinct keys by args
        return pd.DataFrame({"x": range(n)})

    # create two cache entries (two different keys)
    f(3)
    f(5)

    # locate one key dir via manifest, extract the key name (parent dir)
    manifest = next(Path(CACHE).rglob("manifest.json"))
    key_dir = manifest.parent
    qual = f.__qualname__
    before = count_key_dirs(CACHE, qual)

    # remove just that key
    removed = clear(CACHE, qual, key_dir.name)
    after = count_key_dirs(CACHE, qual)

    assert removed == 1
    assert after == before - 1
    # other key is still present
    assert after >= 1

def test_clear_function_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=CACHE, version="v1")
    def g(n: int):
        return {"df": pd.DataFrame({"x": range(n)})}

    g(2)
    g(4)
    qual = g.__qualname__

    # ensure function dir exists with at least one key
    assert count_key_dirs(CACHE, qual) >= 1

    # remove function directory (all keys)
    removed = clear(CACHE, qual)
    assert removed == 1
    # function dir should be gone
    assert not (Path(CACHE) / qual).exists()

def test_clear_all_cache_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=CACHE, version="v1")
    def a():
        return pd.DataFrame({"x": [1, 2]})

    @persist_cache(cache_dir=CACHE, version="v1")
    def b():
        return (pd.DataFrame({"y": [3]}), 42)

    a()
    b()

    # root exists before clear
    assert Path(CACHE).exists()

    removed = clear(CACHE)
    assert removed == 1
    assert not Path(CACHE).exists()

    # after full clear, re-run creates fresh cache
    a()
    manifest, meta = read_manifest(CACHE)
    # basic sanity: manifest is readable after regeneration
    assert isinstance(json.loads(manifest.read_text()), dict)

