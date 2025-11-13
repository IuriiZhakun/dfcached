from dfcached import persist_cache
import time
import threading
from pathlib import Path
import json
import pandas as pd

def test_lock_prevents_double_compute(tmp_path, monkeypatch):
    """
    Two concurrent calls for the same key should compute exactly once.
    The second should load the cached result after the first writes manifest.
    """
    monkeypatch.chdir(tmp_path)

    calls = []

    @persist_cache(cache_dir=".dfcached_cache", version="v1")
    def slow(n: int):
        calls.append(time.time())  # records only if body runs
        time.sleep(0.2)
        df = pd.DataFrame({"x": range(n)})
        return df

    # start two threads simultaneously
    barrier = threading.Barrier(2)

    def worker(results, idx):
        barrier.wait()
        results[idx] = slow(5)

    results = [None, None]
    t1 = threading.Thread(target=worker, args=(results, 0))
    t2 = threading.Thread(target=worker, args=(results, 1))
    t1.start(); t2.start(); t1.join(); t2.join()

    # only one execution of function body
    assert len(calls) == 1
    # both results equal DataFrames
    assert results[0].equals(results[1])

    # sanity: manifest exists
    manifests = list(Path(".dfcached_cache").rglob("manifest.json"))
    assert manifests, "manifest should exist for the cached key"

