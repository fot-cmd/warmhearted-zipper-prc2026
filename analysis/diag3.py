import glob
import os
import numpy as np
import polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(240)

TEAM = "warmhearted-zipper"
SCORES = {6: 315.0646, 7: 314.9031, 8: 317.5528}
TZ = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}
COLS = [
    "MVT_ID_mvt",
    "FLIGHT_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AOBT_3_flt",
    "ARVT_3_flt",
    "AIRCRAFT_TYPE_mvt",
    "TAXITIME_SEC_mvt",
]
M = pl.col("MVT_TIME_UTC_mvt")
B = pl.col("BLOCK_TIME_UTC_mvt")
S = pl.col("SCHED_TIME_UTC_mvt")


def sec(a, b):
    return (a - b).dt.total_seconds().cast(pl.Float64)


def ps_bin(c):
    return (
        pl.when(c.is_null())
        .then(pl.lit("na"))
        .when(c <= 1800)
        .then(pl.lit("a<30m"))
        .when(c <= 3600)
        .then(pl.lit("b30-60m"))
        .when(c <= 7200)
        .then(pl.lit("c1-2h"))
        .when(c <= 14400)
        .then(pl.lit("d2-4h"))
        .when(c <= 57600)
        .then(pl.lit("e4-16h"))
        .otherwise(pl.lit("f>16h"))
    )


def day_cross(f):
    parts = []
    for a, tz in TZ.items():
        x = f.filter(pl.col("ADEP_mvt") == a).with_columns(
            (M.dt.convert_time_zone(tz).dt.date() - S.dt.convert_time_zone(tz).dt.date())
            .dt.total_days()
            .fill_null(0)
            .alias("dcross")
        )
        parts.append(x)
    return pl.concat(parts)


train = pl.concat(
    [pl.read_parquet(f, columns=COLS) for f in sorted(glob.glob("training_*.parquet"))], how="vertical_relaxed"
)
rank = pl.read_parquet("ranking.parquet", columns=COLS)

print("=== A. arrival side: is the copy behaviour the same in 2026? ===")


def arr_stats(f, label):
    a = f.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
        [
            (sec(B, S).abs() <= 6).alias("copy"),
            (sec(B, pl.col("ARVT_3_flt")).abs() <= 6).fill_null(False).alias("copy_arvt"),
            sec(M, S).alias("delay"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("taxi"),
        ]
    )
    return a.group_by("ADES_mvt").agg(
        [
            pl.len().alias(f"n_{label}"),
            pl.col("copy").mean().round(4).alias(f"copy_{label}"),
            pl.col("copy").filter(pl.col("delay") > 3600).mean().round(3).alias(f"copy_late_{label}"),
            pl.col("copy_arvt").mean().round(4).alias(f"arvt_{label}"),
            pl.col("taxi").mean().round(0).alias(f"taxi_{label}"),
            (pl.col("taxi") > 1800).mean().round(4).alias(f"long_{label}"),
        ]
    )


hold = train.filter(M.dt.month().is_in([1, 7]))
t = arr_stats(hold, "h25").join(arr_stats(rank, "r26"), on="ADES_mvt", how="full", coalesce=True).sort("ADES_mvt")
print(t)

print("LIRF arrivals: daily copy rate quantiles, Jan+Jul 2025 vs 2026")
for f, label in [(hold, "2025"), (rank, "2026")]:
    d = (
        f.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
        .with_columns([M.dt.date().alias("day"), (sec(B, S).abs() <= 6).alias("copy")])
        .group_by("day")
        .agg(pl.col("copy").mean())
    )
    q = [round(float(d["copy"].quantile(x)), 3) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
    print("  ", label, "days", d.height, "q10/25/50/75/90", q, "max", round(float(d["copy"].max()), 3))

print("")
print("=== B. departure side: holdout (Jan+Jul 2025) vs ranking (2026) ===")


def dep_frame(f):
    d = f.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        [
            sec(M, S).alias("ps"),
            pl.col("AOBT_3_flt").is_null().alias("an"),
            pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("acn"),
            pl.col("FLIGHT_mvt").str.len_chars().fill_null(0).alias("flen"),
        ]
    )
    return day_cross(d).with_columns(ps_bin(pl.col("ps")).alias("psb"))


hd = dep_frame(hold)
rd = dep_frame(rank)


def dep_stats(d, label):
    return d.group_by("ADEP_mvt").agg(
        [
            pl.len().alias(f"n_{label}"),
            pl.col("an").mean().round(4).alias(f"an_{label}"),
            pl.col("acn").mean().round(4).alias(f"acn_{label}"),
            (pl.col("flen") == 8).mean().round(4).alias(f"fl8_{label}"),
            (pl.col("ps") > 3600).mean().round(4).alias(f"ps1h_{label}"),
            (pl.col("ps") > 14400).mean().round(5).alias(f"ps4h_{label}"),
            (pl.col("dcross") > 0).mean().round(4).alias(f"dx_{label}"),
        ]
    )


print(dep_stats(hd, "h25").join(dep_stats(rd, "r26"), on="ADEP_mvt", how="full", coalesce=True).sort("ADEP_mvt"))

print("LIRF departures without AOBT, by delay bin and day cross: holdout 2025 vs ranking 2026")
g1 = hd.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("an")).group_by(["psb", "dcross"]).agg(
    [
        pl.len().alias("n_h25"),
        (sec(B, S).abs() <= 6).mean().round(3).alias("copy_h25"),
        (pl.col("TAXITIME_SEC_mvt") > 43200).sum().alias("huge_h25"),
    ]
)
g2 = rd.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("an")).group_by(["psb", "dcross"]).agg(pl.len().alias("n_r26"))
print(g1.join(g2, on=["psb", "dcross"], how="full", coalesce=True).sort(["psb", "dcross"]))

print("")
print("=== C. where do our submissions differ? ===")
files = {}
for p in sorted(glob.glob(f"{TEAM}_v*.parquet")):
    try:
        k = int(p.split("_v")[-1].split(".")[0])
    except ValueError:
        continue
    files[k] = pl.read_parquet(p).rename({"TAXITIME_SEC_mvt": f"p{k}"})
print("found versions:", sorted(files))

base = rd.select(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "an", "acn", "flen", "ps", "psb", "dcross"])
for k in sorted(files):
    base = base.join(files[k], on="MVT_ID_mvt", how="left")

pairs = [(a, b) for a, b in [(6, 7), (7, 8), (6, 8), (7, 9), (8, 9), (7, 10)] if a in files and b in files]
for a, b in pairs:
    pa, pb = f"p{a}", f"p{b}"
    x = base.with_columns(((pl.col(pb) - pl.col(pa)).cast(pl.Float64) ** 2).alias("d2"))
    tot = float(x["d2"].sum())
    print("")
    print(f"--- v{a} vs v{b}: rms difference {np.sqrt(tot / x.height):.1f}", end="")
    if a in SCORES and b in SCORES:
        print(f"  score v{a} {SCORES[a]} v{b} {SCORES[b]}", end="")
    print("")
    seg = (
        x.with_columns(
            pl.when(pl.col("ADEP_mvt") == "LIRF").then(pl.lit("LIRF")).otherwise(pl.lit("other")).alias("grp")
        )
        .group_by(["grp", "an", "psb"])
        .agg(
            [
                pl.len().alias("n"),
                (pl.col("d2").sum() / max(tot, 1.0)).round(4).alias("share_diff"),
                pl.col("d2").mean().sqrt().round(1).alias("rms_diff"),
                pl.col(pa).mean().round(0).alias(f"mean_{pa}"),
                pl.col(pb).mean().round(0).alias(f"mean_{pb}"),
            ]
        )
        .sort("share_diff", descending=True)
        .head(15)
    )
    print(seg)
    print("largest 20 differences")
    print(
        x.sort("d2", descending=True)
        .select(["ADEP_mvt", "FLIGHT_mvt", "an", "acn", "flen", "ps", "dcross", pa, pb])
        .head(20)
    )
