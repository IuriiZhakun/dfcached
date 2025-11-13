from dfcached import persist_cache
import pandas as pd, time

@persist_cache(cache_dir=".dfcached_cache", version="v1")
def build(n: int):
    time.sleep(0.2)
    df = pd.DataFrame({"x": range(n), "y": [i*i for i in range(n)]})
    return (df, {"n": n})

def test_hot_hit(tmp_path, monkeypatch):
    # isolate cache in tmp dir so tests are hermetic
    monkeypatch.chdir(tmp_path)
    a1 = build(5)
    a2 = build(5)
    assert a1[0].equals(a2[0]) and a1[1] == a2[1]

