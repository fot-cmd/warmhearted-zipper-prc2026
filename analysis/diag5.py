import glob
import os
import polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

COLS = [
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "STAND_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AOBT_3_flt",
    "TAXITIME_SEC_mvt",
]
M = pl.col("MVT_TIME_UTC_mvt")
B = pl.col("BLOCK_TIME_UTC_mvt")
S = pl.col("SCHED_TIME_UTC_mvt")
MAX_GAP_H = 24


def sec(a, b):
    return (a - b).dt.total_seconds().cast(pl.Float64)


def link(frame, with_labels):
    arr = (
        frame.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & B.is_not_null())
        .select(
            [
                pl.col("ADES_mvt").alias("apt"),
                pl.col("STAND_mvt").alias("stand"),
                B.alias("t"),
                (sec(B, S).abs() <= 6).alias("in_copy"),
                sec(B, S).alias("in_delay"),
            ]
        )
        .sort("t")
    )
    dep = frame.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        [
            pl.col("ADEP_mvt").alias("apt"),
            pl.col("STAND_mvt").alias("stand"),
            S.alias("t"),
            sec(M, S).alias("ps"),
            pl.col("AOBT_3_flt").is_null().alias("an"),
        ]
    )
    if with_labels:
        dep = dep.with_columns(
            [
                (sec(B, S).abs() <= 6).alias("copy"),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
            ]
        )
    dep = dep.filter(pl.col("stand").is_not_null() & pl.col("t").is_not_null()).sort("t")
    j = dep.join_asof(arr.rename({"t": "t_in"}), left_on="t", right_on="t_in", by=["apt", "stand"], strategy="backward")
    j = j.with_columns(sec(pl.col("t"), pl.col("t_in")).alias("gap"))
    j = j.with_columns(
        pl.when(pl.col("t_in").is_null() | (pl.col("gap") > MAX_GAP_H * 3600))
        .then(pl.lit("none"))
        .when(pl.col("in_copy"))
        .then(pl.lit("inbound_copy"))
        .otherwise(pl.lit("inbound_real"))
        .alias("inbound")
    )
    return j


train = pl.concat(
    [pl.read_parquet(f, columns=COLS) for f in sorted(glob.glob("training_*.parquet"))], how="vertical_relaxed"
)
rank = pl.read_parquet("ranking.parquet", columns=COLS)

tr = link(train, True)
rk = link(rank, False)

print("=== departure copy rate by status of the previous arrival at the same stand (2025) ===")
print(
    tr.group_by(["apt", "inbound"])
    .agg([pl.len().alias("n"), pl.col("copy").mean().round(4).alias("dep_copy")])
    .sort(["apt", "inbound"])
)

print("=== same, only delayed departures (ps > 1h), where a copy costs the most ===")
print(
    tr.filter(pl.col("ps") > 3600)
    .group_by(["apt", "an", "inbound"])
    .agg(
        [
            pl.len().alias("n"),
            pl.col("copy").mean().round(4).alias("dep_copy"),
            (pl.col("y") > 3600).mean().round(4).alias("long"),
        ]
    )
    .filter(pl.col("n") >= 20)
    .sort(["apt", "an", "inbound"])
)

print("=== LIRF, by gap between inbound block-in and departure schedule ===")
print(
    tr.filter((pl.col("apt") == "LIRF") & (pl.col("inbound") != "none"))
    .with_columns(
        pl.when(pl.col("gap") <= 3600)
        .then(pl.lit("a<1h"))
        .when(pl.col("gap") <= 3 * 3600)
        .then(pl.lit("b1-3h"))
        .when(pl.col("gap") <= 8 * 3600)
        .then(pl.lit("c3-8h"))
        .otherwise(pl.lit("d>8h"))
        .alias("gapb")
    )
    .group_by(["gapb", "inbound"])
    .agg([pl.len().alias("n"), pl.col("copy").mean().round(4).alias("dep_copy")])
    .sort(["gapb", "inbound"])
)

print("=== coverage in ranking 2026: share of departures with a linked inbound ===")
print(
    rk.group_by("apt")
    .agg(
        [
            pl.len().alias("n"),
            (pl.col("inbound") != "none").mean().round(3).alias("linked"),
            (pl.col("inbound") == "inbound_copy").mean().round(4).alias("inbound_copy"),
        ]
    )
    .sort("apt")
)
