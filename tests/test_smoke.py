from dfcached import persist_cache
import pandas as pd, time

def test_hot_hit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1")
    def build(n: int):
        time.sleep(0.2)
        df = pd.DataFrame({"x": range(n), "y": [i*i for i in range(n)]})
        return (df, {"n": n})

    a1 = build(5)            # cold
    a2 = build(5)            # hot
    assert a1[0].equals(a2[0]) and a1[1] == a2[1]

def test_roundtrip_dict_keys(tmp_path, monkeypatch):
    """Dict keys (ints/tuples) should survive save/load exactly."""
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1")
    def make():
        return {42: "answer", (1, 2): ["ok"]}

    a = make()               # cold
    b = make()               # hot
    assert a == b
    assert 42 in b and (1, 2) in b

def test_refresh_forces_recompute(tmp_path, monkeypatch):
    """refresh=True should recompute every call."""
    monkeypatch.chdir(tmp_path)

    @persist_cache(cache_dir=".dfcached_cache", version="v1", refresh=True)
    def tick():
        return {"t": time.time()}

    x = tick()
    y = tick()
    assert x != y

