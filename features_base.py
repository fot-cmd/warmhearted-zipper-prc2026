import glob
import os
import sys
import numpy as np
import polars as pl

DATA = os.environ.get("DC_DATA", ".")
TEAM = os.environ.get("DC_TEAM", "warmhearted-zipper")
VERSION = os.environ.get("DC_VERSION", "2")

TARGET = "TAXITIME_SEC_mvt"
ANCHOR = "MVT_TIME_UTC_mvt"
AIRPORT = "ADEP_mvt"

Y_MIN = 60
Y_MAX = 5400
BASE_MIN = 60
BASE_MAX = 3600
PRED_MIN = 60
PRED_MAX = 4500

BACK_WINDOWS = [15, 30, 60]
FWD_WINDOWS = [15, 30]
VALID_MONTHS = [1, 7]

N_FOLDS = 5
TE_SMOOTHING = 100.0
SEEDS = [0, 1, 2]
REFIT_SCALE = 1.15

ENCODINGS = [
    (["ADEP_mvt", "RUNWAY_mvt"], "te_runway"),
    (["ADEP_mvt", "STAND_mvt"], "te_stand"),
    (["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], "te_stand_runway"),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], "te_operator"),
    (["ADEP_mvt", "AIRCRAFT_TYPE_mvt"], "te_aircraft"),
    (["ADEP_mvt", "hour"], "te_hour"),
    (["ADEP_mvt", "dow", "hour"], "te_dow_hour"),
]

CAT_COLS = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "stand_runway",
    "AIRCRAFT_TYPE_mvt",
    "MARKET_SEGMENT_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "FLIGHT_RULE_mvt",
    "FLIGHT_TYPE_flt",
    "base_src",
]

PROXY_COLS = [
    "p_aobt",
    "p_lobt",
    "p_eobt",
    "p_iobt",
    "p_sched",
    "aobt_minus_eobt",
    "aobt_minus_iobt",
    "aobt_minus_sched",
    "lobt_minus_iobt",
    "eobt_minus_iobt",
    "sched_minus_iobt",
    "base",
]

TIME_COLS = [
    "hour",
    "minute_of_day",
    "dow",
    "month",
    "day_of_year",
    "week_of_year",
    "tod_sin",
    "tod_cos",
    "doy_sin",
    "doy_cos",
]

OCCUPANCY_COLS = ["taxi_out_active", "taxi_in_active", "runway_out_active"]

GAP_COLS = ["gap_prev_dep", "gap_next_dep", "gap_prev_rwy", "gap_next_rwy"]

TRAFFIC_COLS = (
    [f"dep_back_{w}" for w in BACK_WINDOWS]
    + [f"arr_back_{w}" for w in BACK_WINDOWS]
    + [f"rwy_back_{w}" for w in BACK_WINDOWS]
    + [f"dep_fwd_{w}" for w in FWD_WINDOWS]
    + [f"arr_fwd_{w}" for w in FWD_WINDOWS]
    + [f"rwy_fwd_{w}" for w in FWD_WINDOWS]
    + ["arr_taxi_mean_30", "arr_taxi_mean_60"]
)

TE_COLS = [name for _, name in ENCODINGS]

NUM_COLS = PROXY_COLS + TIME_COLS + OCCUPANCY_COLS + GAP_COLS + TRAFFIC_COLS + TE_COLS
FEATURES = CAT_COLS + NUM_COLS


def load_training():
    files = sorted(glob.glob(os.path.join(DATA, "training_*.parquet")))
    if not files:
        sys.exit("no training parquet files found")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def cumulative_events(frame, key_cols, time_col, value_col=None):
    cols = [pl.col(c) for c in key_cols] + [pl.col(time_col).alias("t")]
    if value_col is not None:
        cols.append(pl.col(value_col).cast(pl.Float64).alias("v"))
    out = frame.select(cols).drop_nulls(["t"] + key_cols).sort(key_cols + ["t"])
    out = out.with_columns(pl.int_range(1, pl.len() + 1).over(key_cols).alias("c"))
    if value_col is not None:
        out = out.with_columns(pl.col("v").fill_null(0).cum_sum().over(key_cols).alias("s"))
    return out


def asof_lookup(left, events, key_cols, take):
    return (
        left.sort("t")
        .join_asof(events.sort("t"), on="t", by=key_cols, strategy="backward")
        .select(["_ridx"] + take)
    )


def window_aggregate(anchors, events, key_cols, minutes, name, forward=False, mean_of=False):
    take = ["c"] + (["s"] if mean_of else [])
    left = anchors.select(["_ridx"] + key_cols + ["t"])
    delta = pl.duration(minutes=minutes)
    upper = left.with_columns((pl.col("t") + delta).alias("t")) if forward else left
    lower = left if forward else left.with_columns((pl.col("t") - delta).alias("t"))

    hi = asof_lookup(upper, events, key_cols, take).rename({c: f"{c}_hi" for c in take})
    lo = asof_lookup(lower, events, key_cols, take).rename({c: f"{c}_lo" for c in take})
    joined = hi.join(lo, on="_ridx", how="full", coalesce=True)

    count = pl.col("c_hi").fill_null(0) - pl.col("c_lo").fill_null(0)
    if mean_of:
        total = pl.col("s_hi").fill_null(0) - pl.col("s_lo").fill_null(0)
        expr = pl.when(count > 0).then(total / count).otherwise(None).alias(name)
    else:
        expr = count.alias(name)
    return joined.select(["_ridx", expr])


def active_count(anchors, starts, ends, key_cols, name):
    left = anchors.select(["_ridx"] + key_cols + ["t"])
    opened = asof_lookup(left, starts, key_cols, ["c"]).rename({"c": "opened"})
    closed = asof_lookup(left, ends, key_cols, ["c"]).rename({"c": "closed"})
    return (
        opened.join(closed, on="_ridx", how="full", coalesce=True)
        .select(
            [
                "_ridx",
                (pl.col("opened").fill_null(0) - pl.col("closed").fill_null(0))
                .clip(0, None)
                .alias(name),
            ]
        )
    )


def departure_spans(frame):
    return frame.select(
        [
            pl.col(AIRPORT).alias("apt"),
            pl.col("RUNWAY_mvt").alias("rwy"),
            pl.coalesce([pl.col("AOBT_3_flt"), pl.col("LOBT_flt"), pl.col("EOBT_1_flt")]).alias("start"),
            pl.col(ANCHOR).alias("end"),
        ]
    ).drop_nulls(["start", "end"]).filter(pl.col("end") > pl.col("start"))


def arrival_spans(frame):
    return frame.select(
        [
            pl.col("ADES_mvt").alias("apt"),
            pl.col(ANCHOR).alias("start"),
            (pl.col(ANCHOR) + pl.duration(seconds=pl.col(TARGET).cast(pl.Int64))).alias("end"),
        ]
    ).drop_nulls(["start", "end"]).filter(pl.col("end") > pl.col("start"))


def add_gaps(dep):
    dep = dep.sort(["apt_key", "t_key"])
    dep = dep.with_columns(
        [
            (pl.col("t_key") - pl.col("t_key").shift(1).over("apt_key")).dt.total_seconds().alias("gap_prev_dep"),
            (pl.col("t_key").shift(-1).over("apt_key") - pl.col("t_key")).dt.total_seconds().alias("gap_next_dep"),
        ]
    )
    dep = dep.sort(["apt_key", "rwy_key", "t_key"])
    dep = dep.with_columns(
        [
            (pl.col("t_key") - pl.col("t_key").shift(1).over(["apt_key", "rwy_key"])).dt.total_seconds().alias("gap_prev_rwy"),
            (pl.col("t_key").shift(-1).over(["apt_key", "rwy_key"]) - pl.col("t_key")).dt.total_seconds().alias("gap_next_rwy"),
        ]
    )
    return dep


def add_context(dep, full):
    dep = dep.with_columns(
        [
            pl.col(AIRPORT).alias("apt_key"),
            pl.col("RUNWAY_mvt").alias("rwy_key"),
            pl.col(ANCHOR).alias("t_key"),
        ]
    )
    dep = add_gaps(dep).with_row_index("_ridx")

    anchors = dep.select(
        [
            "_ridx",
            pl.col("apt_key").alias("apt"),
            pl.col("rwy_key").alias("rwy"),
            pl.col("t_key").alias("t"),
        ]
    )

    dep_rows = full.filter(pl.col("PHASE_mvt") == "DEP")
    arr_rows = full.filter(pl.col("PHASE_mvt") == "ARR")

    dep_events = cumulative_events(
        dep_rows.select([pl.col(AIRPORT).alias("apt"), ANCHOR]), ["apt"], ANCHOR
    )
    rwy_events = cumulative_events(
        dep_rows.select([pl.col(AIRPORT).alias("apt"), pl.col("RUNWAY_mvt").alias("rwy"), ANCHOR]),
        ["apt", "rwy"],
        ANCHOR,
    )
    arr_events = cumulative_events(
        arr_rows.select([pl.col("ADES_mvt").alias("apt"), ANCHOR]), ["apt"], ANCHOR
    )
    arr_taxi_events = cumulative_events(
        arr_rows.select([pl.col("ADES_mvt").alias("apt"), ANCHOR, TARGET]),
        ["apt"],
        ANCHOR,
        value_col=TARGET,
    )

    dspan = departure_spans(dep_rows)
    aspan = arrival_spans(arr_rows)

    parts = []
    for w in BACK_WINDOWS:
        parts.append(window_aggregate(anchors, dep_events, ["apt"], w, f"dep_back_{w}"))
        parts.append(window_aggregate(anchors, arr_events, ["apt"], w, f"arr_back_{w}"))
        parts.append(window_aggregate(anchors, rwy_events, ["apt", "rwy"], w, f"rwy_back_{w}"))
    for w in FWD_WINDOWS:
        parts.append(window_aggregate(anchors, dep_events, ["apt"], w, f"dep_fwd_{w}", forward=True))
        parts.append(window_aggregate(anchors, arr_events, ["apt"], w, f"arr_fwd_{w}", forward=True))
        parts.append(window_aggregate(anchors, rwy_events, ["apt", "rwy"], w, f"rwy_fwd_{w}", forward=True))
    for w in [30, 60]:
        parts.append(
            window_aggregate(anchors, arr_taxi_events, ["apt"], w, f"arr_taxi_mean_{w}", mean_of=True)
        )

    parts.append(
        active_count(
            anchors,
            cumulative_events(dspan, ["apt"], "start"),
            cumulative_events(dspan, ["apt"], "end"),
            ["apt"],
            "taxi_out_active",
        )
    )
    parts.append(
        active_count(
            anchors,
            cumulative_events(aspan, ["apt"], "start"),
            cumulative_events(aspan, ["apt"], "end"),
            ["apt"],
            "taxi_in_active",
        )
    )
    parts.append(
        active_count(
            anchors,
            cumulative_events(dspan, ["apt", "rwy"], "start"),
            cumulative_events(dspan, ["apt", "rwy"], "end"),
            ["apt", "rwy"],
            "runway_out_active",
        )
    )

    for p in parts:
        dep = dep.join(p, on="_ridx", how="left")
    return dep.drop(["_ridx", "apt_key", "rwy_key", "t_key"])


def add_features(dep):
    t = pl.col(ANCHOR)
    dep = dep.with_columns(
        [
            (t - pl.col("AOBT_3_flt")).dt.total_seconds().alias("p_aobt"),
            (t - pl.col("LOBT_flt")).dt.total_seconds().alias("p_lobt"),
            (t - pl.col("EOBT_1_flt")).dt.total_seconds().alias("p_eobt"),
            (t - pl.col("IOBT_flt")).dt.total_seconds().alias("p_iobt"),
            (t - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("p_sched"),
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("aobt_minus_eobt"),
            (pl.col("AOBT_3_flt") - pl.col("IOBT_flt")).dt.total_seconds().alias("aobt_minus_iobt"),
            (pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("aobt_minus_sched"),
            (pl.col("LOBT_flt") - pl.col("IOBT_flt")).dt.total_seconds().alias("lobt_minus_iobt"),
            (pl.col("EOBT_1_flt") - pl.col("IOBT_flt")).dt.total_seconds().alias("eobt_minus_iobt"),
            (pl.col("SCHED_TIME_UTC_mvt") - pl.col("IOBT_flt")).dt.total_seconds().alias("sched_minus_iobt"),
            t.dt.hour().alias("hour"),
            (t.dt.hour() * 60 + t.dt.minute()).alias("minute_of_day"),
            t.dt.weekday().alias("dow"),
            t.dt.month().alias("month"),
            t.dt.ordinal_day().alias("day_of_year"),
            t.dt.week().alias("week_of_year"),
            pl.concat_str([pl.col("STAND_mvt"), pl.col("RUNWAY_mvt")], separator="|", ignore_nulls=False).alias("stand_runway"),
        ]
    )
    dep = dep.with_columns(
        [
            (2 * np.pi * pl.col("minute_of_day") / 1440).sin().alias("tod_sin"),
            (2 * np.pi * pl.col("minute_of_day") / 1440).cos().alias("tod_cos"),
            (2 * np.pi * pl.col("day_of_year") / 365.25).sin().alias("doy_sin"),
            (2 * np.pi * pl.col("day_of_year") / 365.25).cos().alias("doy_cos"),
        ]
    )
    dep = dep.with_columns(
        pl.when(pl.col("p_aobt").is_between(BASE_MIN, BASE_MAX))
        .then(pl.lit("aobt"))
        .when(pl.col("p_lobt").is_between(BASE_MIN, BASE_MAX))
        .then(pl.lit("lobt"))
        .when(pl.col("p_eobt").is_between(BASE_MIN, BASE_MAX))
        .then(pl.lit("eobt"))
        .when(pl.col("p_iobt").is_between(BASE_MIN, BASE_MAX))
        .then(pl.lit("iobt"))
        .otherwise(pl.lit("none"))
        .alias("base_src")
    )
    dep = dep.with_columns(
        pl.coalesce(
            [
                pl.col("p_aobt").clip(BASE_MIN, BASE_MAX),
                pl.col("p_lobt").clip(BASE_MIN, BASE_MAX),
                pl.col("p_eobt").clip(BASE_MIN, BASE_MAX),
                pl.col("p_iobt").clip(BASE_MIN, BASE_MAX),
                pl.lit(900.0),
            ]
        ).alias("base")
    )
    return dep


def prepare(frame, full):
    frame = add_context(frame, full)
    frame = add_features(frame)
    return frame


