import glob
import polars as pl

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

COLS = [
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AOBT_3_flt",
    "EOBT_1_flt",
    "IOBT_flt",
    "ARVT_3_flt",
    "TAXITIME_SEC_mvt",
    "STAND_mvt",
    "RUNWAY_mvt",
    "FLIGHT_mvt",
]

files = sorted(glob.glob("training_*.parquet"))
df = pl.concat([pl.read_parquet(f, columns=COLS) for f in files], how="vertical_relaxed")

b = pl.col("BLOCK_TIME_UTC_mvt")
m = pl.col("MVT_TIME_UTC_mvt")


def near(c):
    return ((b - pl.col(c)).dt.total_seconds().abs() <= 6).fill_null(False)


df = df.with_columns(
    [
        pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt"),
        m.dt.date().alias("day"),
        m.dt.truncate("3h").alias("slot"),
        m.dt.month().alias("month"),
        near("SCHED_TIME_UTC_mvt").alias("sub_sched"),
        (near("SCHED_TIME_UTC_mvt") | near("AOBT_3_flt") | near("EOBT_1_flt") | near("IOBT_flt")).alias("matched"),
        ((b - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() / 60).round().alias("bs_min"),
        ((b - pl.col("AOBT_3_flt")).dt.total_seconds() / 60).round().alias("ba_min"),
        ((m - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() / 60).round().alias("ms_min"),
        (pl.col("TAXITIME_SEC_mvt") > 3600).alias("long"),
        b.dt.second().alias("b_sec"),
    ]
)

dep = df.filter(pl.col("PHASE_mvt") == "DEP")
arr = df.filter(pl.col("PHASE_mvt") == "ARR")

print("=== 1. ARR vs DEP sched-copy correlation ===")
for key in ["day", "slot"]:
    dd = dep.group_by(["apt", key]).agg(
        [
            pl.len().alias("dn"),
            pl.col("sub_sched").mean().alias("dep_sub"),
            pl.col("long").mean().alias("dep_long"),
        ]
    )
    ad = arr.group_by(["apt", key]).agg(
        [
            pl.len().alias("an"),
            pl.col("sub_sched").mean().alias("arr_sub"),
            pl.col("long").mean().alias("arr_long"),
        ]
    )
    j = dd.join(ad, on=["apt", key], how="inner").filter((pl.col("dn") >= 5) & (pl.col("an") >= 5))
    print("granularity:", key)
    print(
        j.group_by("apt")
        .agg(
            [
                pl.len().alias("periods"),
                pl.corr("dep_sub", "arr_sub").alias("corr_sub"),
                pl.corr("dep_long", "arr_long").alias("corr_long"),
                pl.col("dep_sub").mean().alias("dep_sub_mean"),
                pl.col("arr_sub").mean().alias("arr_sub_mean"),
            ]
        )
        .sort("apt")
    )

print("=== 2. squared error share of long rows by airport (vs airport mean) ===")
tot = dep.select(((pl.col("TAXITIME_SEC_mvt") - pl.col("TAXITIME_SEC_mvt").mean()) ** 2).sum()).item()
print(
    dep.group_by("apt")
    .agg(
        [
            pl.len().alias("n"),
            pl.col("long").sum().alias("n_long"),
            (pl.col("long") & pl.col("matched")).sum().alias("long_matched"),
            (pl.col("long") & ~pl.col("matched")).sum().alias("long_unmatched"),
            (((pl.col("TAXITIME_SEC_mvt") - 1000) ** 2).filter(pl.col("long")).sum() / tot).alias("ss_share_long"),
        ]
    )
    .sort("apt")
)

for a in ["LIRF", "LFPG"]:
    x = dep.filter((pl.col("apt") == a) & pl.col("long") & ~pl.col("matched"))
    print(f"=== 3. {a} long unmatched rows: {x.height} ===")
    print("by month")
    print(x.group_by("month").len().sort("month").transpose(include_header=True))
    print("top days")
    print(x.group_by("day").len().sort("len", descending=True).head(12))
    print("summary")
    print(
        x.select(
            [
                pl.col("AOBT_3_flt").is_null().mean().alias("aobt_null"),
                pl.col("EOBT_1_flt").is_null().mean().alias("eobt_null"),
                pl.col("STAND_mvt").is_null().mean().alias("stand_null"),
                (pl.col("b_sec") < 6).mean().alias("block_whole_min"),
                pl.col("bs_min").median().alias("bs_med"),
                pl.col("ba_min").median().alias("ba_med"),
                pl.col("ms_min").median().alias("ms_med"),
                pl.col("TAXITIME_SEC_mvt").median().alias("y_med"),
            ]
        )
    )
    print("BLOCK minus SCHED (minutes), most common")
    print(x.group_by("bs_min").len().sort("len", descending=True).head(12).transpose(include_header=True))
    print("top stands")
    print(x.group_by("STAND_mvt").len().sort("len", descending=True).head(10).transpose(include_header=True))
    print("samples")
    print(
        x.sample(min(15, x.height), seed=0).select(
            [
                "FLIGHT_mvt",
                "STAND_mvt",
                "SCHED_TIME_UTC_mvt",
                "BLOCK_TIME_UTC_mvt",
                "AOBT_3_flt",
                "EOBT_1_flt",
                "MVT_TIME_UTC_mvt",
                "TAXITIME_SEC_mvt",
            ]
        )
    )
