import os
import numpy as np
import polars as pl

DATA = os.environ.get("DC_DATA", ".")
TEAM = os.environ.get("DC_TEAM", "warmhearted-zipper")
VERSION = os.environ.get("DC_VERSION", "11")
TARGET = "TAXITIME_SEC_mvt"
VARIANTS = [v.strip() for v in os.environ.get("DC_VARIANTS", "c6,c7").split(",") if v.strip()]
WRITE_FINAL = os.environ.get("DC_WRITE_FINAL", "1") == "1"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


print("variants:", VARIANTS)
hold = {}
y = None
for v in VARIANTS:
    path = f"holdout_{v[0]}.npz"
    if os.path.exists(path):
        h = np.load(path)
        if v in h.files:
            if y is None:
                y = h["y"]
            elif len(h["y"]) != len(y) or not np.allclose(h["y"], y):
                print("holdout rows differ for", v, "- holdout comparison skipped for it")
                continue
            hold[v] = h[v]
if hold:
    print("holdout rmse per variant:")
    for v in VARIANTS:
        if v in hold:
            print(f"  {v}: {rmse(hold[v], y):.2f}")
    if len(hold) == len(VARIANTS):
        avg = np.mean([hold[v] for v in VARIANTS], axis=0)
        print(f"  ensemble: {rmse(avg, y):.2f}")
    print("")


def combine(tag, sub_file, name):
    paths = {v: f"pred_{v}_{tag}.parquet" for v in VARIANTS}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print(tag, "skipped, missing:", missing)
        return
    sub = pl.read_parquet(os.path.join(DATA, sub_file))
    frame = sub.select("MVT_ID_mvt")
    for v, p in paths.items():
        frame = frame.join(pl.read_parquet(p).rename({"pred": v}), on="MVT_ID_mvt", how="left")
    assert frame.height == sub.height, (frame.height, sub.height)
    for v in VARIANTS:
        assert frame[v].null_count() == 0, (tag, v)
    values = np.mean([frame[v].to_numpy() for v in VARIANTS], axis=0)
    print(tag, "rms difference to ensemble per variant:")
    for v in VARIANTS:
        print(f"  {v}: {np.sqrt(np.mean((frame[v].to_numpy() - values) ** 2)):.1f}")
    out = frame.select(
        [
            pl.col("MVT_ID_mvt"),
            pl.Series(TARGET, np.round(values).astype(np.int32)),
        ]
    )
    assert out.schema == sub.schema, (out.schema, sub.schema)
    assert out[TARGET].null_count() == 0
    out.write_parquet(name)
    print("written:", name, out.shape)
    print(out[TARGET].describe())
    print("")


combine("ranking", "submitting.parquet", f"{TEAM}_v{VERSION}.parquet")
if WRITE_FINAL:
    combine("final", "final_submitting.parquet", f"{TEAM}_final.parquet")
