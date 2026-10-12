import glob
import polars as pl

pl.Config.set_tbl_rows(150)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

COLS = [
    "ADEP_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AOBT_3_flt",
    "EOBT_1_flt",
    "TAXITIME_SEC_mvt",
]
BREAKS = [3600, 7200, 14400, 28800, 57600]
LABELS = ["a<1h", "b1-2h", "c2-4h", "d4-8h", "e8-16h", "f>16h"]


def prep(frame):
    m = pl.col("MVT_TIME_UTC_mvt")
    return (
        frame.filter(pl.col("PHASE_mvt") == "DEP")
        .with_columns(
            [
                (m - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("ps"),
                pl.col("AOBT_3_flt").is_null().alias("aobt_null"),
            ]
        )
        .filter(pl.col("ps") > 3600)
        .with_columns(pl.col("ps").cut(BREAKS, labels=LABELS).cast(pl.String).alias("bin"))
    )


files = sorted(glob.glob("training_*.parquet"))
tr = prep(pl.concat([pl.read_parquet(f, columns=COLS) for f in files], how="vertical_relaxed"))
rk = prep(pl.read_parquet("ranking.parquet", columns=COLS))

y = pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)
b = pl.col("BLOCK_TIME_UTC_mvt")
tr = tr.with_columns(
    [
        ((b - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6).fill_null(False).alias("copy"),
        ((b - pl.col("AOBT_3_flt")).dt.total_seconds().abs() <= 6).fill_null(False).alias("copy_aobt"),
        ((b - pl.col("EOBT_1_flt")).dt.total_seconds().abs() <= 6).fill_null(False).alias("copy_eobt"),
    ]
).with_columns(
    [
        ((y > 43200) & ~pl.col("copy")).alias("dayshift"),
        (y <= 3600).alias("normal"),
    ]
)

keys = ["ADEP_mvt", "aobt_null", "bin"]
stats = tr.group_by(keys).agg(
    [
        pl.len().alias("n_tr"),
        pl.col("copy").mean().round(3).alias("copy"),
        pl.col("dayshift").mean().round(3).alias("dayshift"),
        pl.col("normal").mean().round(3).alias("normal"),
        pl.col("copy_aobt").mean().round(3).alias("c_aobt"),
        pl.col("copy_eobt").mean().round(3).alias("c_eobt"),
        y.mean().round(0).alias("y_mean"),
        pl.col("ps").mean().round(0).alias("ps_mean"),
        (y.mean() / pl.col("ps").mean()).round(3).alias("ratio"),
        y.std().round(0).alias("y_std"),
    ]
)
rcount = rk.group_by(keys).agg(pl.len().alias("n_rank"))
out = stats.join(rcount, on=keys, how="full", coalesce=True).sort(keys)

print("=== rows with MVT more than 1h after schedule ===")
print("copy = BLOCK is schedule copy, dayshift = 24h shifted, normal = taxi <= 1h")
print("ratio = mean taxi / mean schedule delay (1.0 means taxi equals the delay)")
print(out.filter((pl.col("n_tr").fill_null(0) >= 10) | (pl.col("n_rank").fill_null(0) >= 10)))

print("=== all airports combined ===")
print(
    tr.group_by(["aobt_null", "bin"])
    .agg(
        [
            pl.len().alias("n_tr"),
            pl.col("copy").mean().round(3).alias("copy"),
            pl.col("dayshift").mean().round(3).alias("dayshift"),
            pl.col("normal").mean().round(3).alias("normal"),
            (y.mean() / pl.col("ps").mean()).round(3).alias("ratio"),
        ]
    )
    .join(rk.group_by(["aobt_null", "bin"]).agg(pl.len().alias("n_rank")), on=["aobt_null", "bin"], how="full", coalesce=True)
    .sort(["aobt_null", "bin"])
)
