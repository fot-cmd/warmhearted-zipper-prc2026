import glob
import os
import sys
import time
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb
import features_base as v2

DATA = os.environ.get("DC_DATA", ".")
TEAM = os.environ.get("DC_TEAM", "warmhearted-zipper")
VERSION = os.environ.get("DC_VERSION", "11")
SKIP_VALID = os.environ.get("DC_SKIP_VALID", "0") == "1"
THREADS = int(os.environ.get("DC_THREADS", "0"))
SEEDS = [int(s) for s in os.environ.get("DC_SEEDS", "0").split(",")]

TARGET = "TAXITIME_SEC_mvt"
JITTER = 6
DAY = 86400.0
DAYSHIFT_MIN = 43200
SPEC_MIN = 3600.0
Y_CLIP = 10800
VALID_MONTHS = [1, 7]
REFIT_SCALE = 1.1
FALLBACK_CLF_ROUNDS = 100
FALLBACK_REG_ROUNDS = 450
FALLBACK_SPEC_ROUNDS = 400
FALLBACK_LOCAL_ROUNDS = 200
FALLBACK_SPEC_WEIGHT = 0.5
FALLBACK_RULE_WEIGHT = 0.9
FALLBACK_ALPHA = 0.5
PRED_MIN = 30
PRED_MAX = 200000
N_CLASSES = 6
TE_SMOOTH = 30.0
TE_FOLDS = 5
RULE_BREAKS = np.array([3600, 7200, 14400, 28800, 57600], dtype=float)
RULE_LABELS = ["<1h", "1-2h", "2-4h", "4-8h", "8-16h", ">16h"]
RULE_MIN_N = 5
RULE_MIN_RATE = 0.2

NEED = [
    "MVT_ID_mvt",
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",
    "LOBT_flt",
    "MARKET_SEGMENT_flt",
    "IOBT_flt",
    "FLIGHT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "EOBT_1_flt",
    "AOBT_3_flt",
    "ARVT_1_flt",
    "ARVT_3_flt",
]

CAT = [
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
    "flt_prefix",
]

NUM_BASE = [
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
    "hour",
    "minute_of_day",
    "dow",
    "month",
    "day_of_year",
    "tod_sin",
    "tod_cos",
    "taxi_out_active",
    "taxi_in_active",
    "runway_out_active",
    "gap_prev_dep",
    "gap_next_dep",
    "gap_prev_rwy",
    "gap_next_rwy",
    "dep_back_15",
    "dep_back_30",
    "dep_back_60",
    "arr_back_15",
    "arr_back_30",
    "arr_back_60",
    "rwy_back_15",
    "rwy_back_30",
    "rwy_back_60",
    "dep_fwd_15",
    "dep_fwd_30",
    "arr_fwd_15",
    "arr_fwd_30",
    "rwy_fwd_15",
    "rwy_fwd_30",
    "arr_taxi_mean_30",
    "arr_taxi_mean_60",
    "mvt_sec",
    "mvt_whole",
    "aobt_null",
    "eobt_null",
    "sched_m5",
    "eobt_minus_sched",
    "lobt_minus_sched",
    "flight_time",
    "arvt_diff",
]

PERIODS = ["day", "3h"]
PERIOD_NAMES = [
    "arr_n",
    "arr_copy",
    "arr_copy_arvt",
    "arr_taxi_mean",
    "arr_taxi_q90",
    "arr_long",
    "dep_n",
    "dep_aobt_null",
    "dep_paobt_med",
    "dep_psched_med",
    "dep_aobt_eobt_med",
]
PERIOD_COLS = [f"{n}_{p}" for p in PERIODS for n in PERIOD_NAMES]

TE_SPECS = [
    (["ADEP_mvt", "FLIGHT_mvt"], "flight"),
    (["ADEP_mvt", "STAND_mvt"], "stand"),
    (["ADEP_mvt", "flt_prefix"], "prefix"),
]
TE_TARGETS = ["t_sched", "t_copy"]
TE_COLS = [f"te_{n}_{t}" for _, n in TE_SPECS for t in TE_TARGETS]
KEY_COLS = ["ADEP_mvt", "FLIGHT_mvt", "STAND_mvt", "flt_prefix"]

INBOUND_COLS = ["in_copy", "in_gap_h", "in_delay", "in_copy_prev2"]
INBOUND_MAX_GAP = 24 * 3600
NUM = NUM_BASE + PERIOD_COLS + INBOUND_COLS
FEAT = CAT + NUM
FEAT_ALL = FEAT + TE_COLS
CAT_IDX = [FEAT_ALL.index(c) for c in CAT]
CLASS_NAMES = ["model", "sched", "aobt", "eobt", "lobt_iobt", "dayshift"]

LOCAL_FEATS = [
    "p_sched",
    "hour",
    "minute_of_day",
    "dow",
    "mvt_sec",
    "mvt_whole",
    "sched_m5",
    "eobt_null",
    "flight_time",
    "arvt_diff",
    "arr_copy_day",
    "arr_copy_3h",
    "arr_copy_arvt_day",
    "arr_n_3h",
    "dep_n_3h",
    "dep_aobt_null_day",
    "dep_aobt_null_3h",
    "dep_psched_med_day",
    "dep_psched_med_3h",
    "te_flight_t_sched",
    "te_flight_t_copy",
    "te_stand_t_sched",
    "te_stand_t_copy",
    "te_prefix_t_sched",
    "te_prefix_t_copy",
    "in_copy",
    "in_gap_h",
    "in_copy_prev2",
]
LOCAL_IDX = [FEAT_ALL.index(c) for c in LOCAL_FEATS]


def read_fixed(path):
    table = pq.read_table(path)
    cols = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if isinstance(field.type, pa.ExtensionType):
            col = pa.chunked_array([ch.storage for ch in col.chunks], type=field.type.storage_type)
        cols.append(col)
    frame = pl.from_arrow(pa.table(cols, names=table.column_names))
    return frame.select(NEED).with_columns(
        pl.col("AIRCRAFT_OPERATOR_flt").cast(pl.String).str.slice(0, 12)
    )


def add_extra(frame):
    m = pl.col("MVT_TIME_UTC_mvt")
    return frame.with_columns(
        [
            m.dt.second().alias("mvt_sec"),
            (m.dt.second() < 6).cast(pl.Int8).alias("mvt_whole"),
            pl.col("AOBT_3_flt").is_null().cast(pl.Int8).alias("aobt_null"),
            pl.col("EOBT_1_flt").is_null().cast(pl.Int8).alias("eobt_null"),
            (pl.col("SCHED_TIME_UTC_mvt").dt.minute() % 5).alias("sched_m5"),
            (pl.col("EOBT_1_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("eobt_minus_sched"),
            (pl.col("LOBT_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("lobt_minus_sched"),
            (pl.col("ARVT_3_flt") - m).dt.total_seconds().alias("flight_time"),
            (pl.col("ARVT_3_flt") - pl.col("ARVT_1_flt")).dt.total_seconds().alias("arvt_diff"),
            pl.col("FLIGHT_mvt").str.slice(0, 3).alias("flt_prefix"),
        ]
    )


def period_key(name):
    m = pl.col("MVT_TIME_UTC_mvt")
    if name == "day":
        return m.dt.date().alias("_k")
    return m.dt.truncate("3h").alias("_k")


def period_tables(ctx, apt, name):
    b = pl.col("BLOCK_TIME_UTC_mvt")
    m = pl.col("MVT_TIME_UTC_mvt")
    t = pl.col(TARGET).cast(pl.Float64)
    arr = (
        ctx.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == apt))
        .with_columns(period_key(name))
        .group_by("_k")
        .agg(
            [
                pl.len().cast(pl.Float32).alias(f"arr_n_{name}"),
                ((b - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= JITTER)
                .mean()
                .cast(pl.Float32)
                .alias(f"arr_copy_{name}"),
                ((b - pl.col("ARVT_3_flt")).dt.total_seconds().abs() <= JITTER)
                .mean()
                .cast(pl.Float32)
                .alias(f"arr_copy_arvt_{name}"),
                t.mean().cast(pl.Float32).alias(f"arr_taxi_mean_{name}"),
                t.quantile(0.9).cast(pl.Float32).alias(f"arr_taxi_q90_{name}"),
                (t > 1800).mean().cast(pl.Float32).alias(f"arr_long_{name}"),
            ]
        )
    )
    dep = (
        ctx.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == apt))
        .with_columns(period_key(name))
        .group_by("_k")
        .agg(
            [
                pl.len().cast(pl.Float32).alias(f"dep_n_{name}"),
                pl.col("AOBT_3_flt").is_null().mean().cast(pl.Float32).alias(f"dep_aobt_null_{name}"),
                (m - pl.col("AOBT_3_flt")).dt.total_seconds().median().cast(pl.Float32).alias(f"dep_paobt_med_{name}"),
                (m - pl.col("SCHED_TIME_UTC_mvt"))
                .dt.total_seconds()
                .median()
                .cast(pl.Float32)
                .alias(f"dep_psched_med_{name}"),
                (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt"))
                .dt.total_seconds()
                .median()
                .cast(pl.Float32)
                .alias(f"dep_aobt_eobt_med_{name}"),
            ]
        )
    )
    return arr, dep


def add_period_context(part, ctx, apt):
    for name in PERIODS:
        arr, dep = period_tables(ctx, apt, name)
        part = (
            part.with_columns(period_key(name))
            .join(arr, on="_k", how="left")
            .join(dep, on="_k", how="left")
            .drop("_k")
        )
    return part


def add_inbound(part, ctx, apt):
    b = pl.col("BLOCK_TIME_UTC_mvt")
    arr = (
        ctx.filter(
            (pl.col("PHASE_mvt") == "ARR")
            & (pl.col("ADES_mvt") == apt)
            & pl.col("STAND_mvt").is_not_null()
            & b.is_not_null()
        )
        .select(
            [
                pl.col("STAND_mvt"),
                b.alias("_t_in"),
                ((b - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= JITTER).cast(pl.Float32).alias("_c"),
                (b - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float32).alias("_d"),
            ]
        )
        .sort(["STAND_mvt", "_t_in"])
        .with_columns(pl.col("_c").shift(1).over("STAND_mvt").alias("_c_prev"))
        .with_columns(((pl.col("_c") + pl.col("_c_prev").fill_null(pl.col("_c"))) / 2).alias("_c2"))
        .sort("_t_in")
    )
    left = (
        part.select(["STAND_mvt", "SCHED_TIME_UTC_mvt"])
        .with_row_index("_ri")
        .filter(pl.col("STAND_mvt").is_not_null() & pl.col("SCHED_TIME_UTC_mvt").is_not_null())
        .sort("SCHED_TIME_UTC_mvt")
    )
    j = left.join_asof(arr, left_on="SCHED_TIME_UTC_mvt", right_on="_t_in", by="STAND_mvt", strategy="backward")
    j = j.with_columns(
        (pl.col("SCHED_TIME_UTC_mvt") - pl.col("_t_in")).dt.total_seconds().cast(pl.Float32).alias("_gap")
    )
    ok = pl.col("_t_in").is_not_null() & (pl.col("_gap") <= INBOUND_MAX_GAP)
    j = j.select(
        [
            "_ri",
            pl.when(ok).then(pl.col("_c")).otherwise(None).alias("in_copy"),
            pl.when(ok).then(pl.col("_gap") / 3600.0).otherwise(None).alias("in_gap_h"),
            pl.when(ok).then(pl.col("_d")).otherwise(None).alias("in_delay"),
            pl.when(ok).then(pl.col("_c2")).otherwise(None).alias("in_copy_prev2"),
        ]
    )
    out = part.with_row_index("_ri").join(j, on="_ri", how="left").sort("_ri").drop("_ri")
    assert out.height == part.height
    return out


def build(frame):
    deps = frame.filter(pl.col("PHASE_mvt") == "DEP")
    airports = sorted(deps["ADEP_mvt"].drop_nulls().unique().to_list())
    parts = []
    for apt in airports:
        ctx = frame.filter((pl.col("ADEP_mvt") == apt) | (pl.col("ADES_mvt") == apt))
        part = v2.prepare(
            ctx.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == apt)), ctx
        )
        part = add_extra(part)
        part = add_period_context(part, ctx, apt)
        part = add_inbound(part, ctx, apt)
        part = part.with_columns(
            [
                pl.col(c).cast(pl.Float32)
                for c, t in part.schema.items()
                if t == pl.Float64 and c != "MVT_ID_mvt"
            ]
        )
        parts.append(part)
        print("  ", apt, part.height, flush=True)
    return pl.concat(parts, how="vertical_relaxed")


def add_labels(frame):
    b = pl.col("BLOCK_TIME_UTC_mvt")

    def near(c):
        return (b - pl.col(c)).dt.total_seconds().abs() <= JITTER

    return frame.with_columns(
        pl.when(near("SCHED_TIME_UTC_mvt"))
        .then(1)
        .when(near("AOBT_3_flt"))
        .then(2)
        .when(near("EOBT_1_flt"))
        .then(3)
        .when(near("LOBT_flt") | near("IOBT_flt"))
        .then(4)
        .when(pl.col(TARGET) > DAYSHIFT_MIN)
        .then(5)
        .otherwise(0)
        .fill_null(0)
        .cast(pl.Int8)
        .alias("cls")
    )


def candidates(frame):
    return np.stack(
        [
            frame["p_sched"].cast(pl.Float64).to_numpy(),
            frame["p_aobt"].cast(pl.Float64).to_numpy(),
            frame["p_eobt"].cast(pl.Float64).to_numpy(),
            frame["p_lobt"].fill_null(frame["p_iobt"]).cast(pl.Float64).to_numpy(),
        ],
        axis=1,
    )


def category_maps(frame):
    maps = {}
    for c in CAT:
        vc = frame[c].cast(pl.String).drop_nulls().value_counts().filter(pl.col("count") >= 20)
        maps[c] = {v: i for i, v in enumerate(vc[c].to_list())}
    return maps


def matrix(frame, maps):
    x = np.empty((frame.height, len(FEAT)), dtype=np.float32)
    for j, c in enumerate(FEAT):
        if c in maps:
            codes = (
                frame[c]
                .cast(pl.String)
                .replace_strict(maps[c], default=-1, return_dtype=pl.Int32)
                .to_numpy()
                .astype(np.float32)
            )
            codes[codes < 0] = np.nan
            x[:, j] = codes
        else:
            x[:, j] = frame[c].cast(pl.Float32).to_numpy()
    return x


def te_stats(fit, keys, target, prior):
    return (
        fit.group_by(keys)
        .agg([pl.col(target).sum().alias("_s"), pl.len().alias("_n")])
        .with_columns(((pl.col("_s") + prior * TE_SMOOTH) / (pl.col("_n") + TE_SMOOTH)).alias("_te"))
        .select(keys + ["_te"])
    )


def te_apply(frame, stats, keys, prior):
    out = frame.select(keys).with_row_index("_i").join(stats, on=keys, how="left").sort("_i")
    return out["_te"].fill_null(prior).to_numpy().astype(np.float32)


def te_oof(fit):
    rng = np.random.default_rng(0)
    folds = rng.integers(0, TE_FOLDS, fit.height)
    out = np.zeros((fit.height, len(TE_COLS)), dtype=np.float32)
    j = 0
    for keys, _ in TE_SPECS:
        for t in TE_TARGETS:
            prior = float(fit[t].mean())
            col = np.empty(fit.height, dtype=np.float32)
            for f in range(TE_FOLDS):
                mask = folds == f
                stats = te_stats(fit.filter(pl.Series(~mask)), keys, t, prior)
                col[mask] = te_apply(fit.filter(pl.Series(mask)), stats, keys, prior)
            out[:, j] = col
            j += 1
    return out


def te_full(fit, other):
    out = np.zeros((other.height, len(TE_COLS)), dtype=np.float32)
    j = 0
    for keys, _ in TE_SPECS:
        for t in TE_TARGETS:
            prior = float(fit[t].mean())
            out[:, j] = te_apply(other, te_stats(fit, keys, t, prior), keys, prior)
            j += 1
    return out


def rule_bins(ps):
    return np.digitize(np.nan_to_num(ps, nan=-1.0), RULE_BREAKS)


def rule_table(apt, anull, ps, cls, y):
    bins = rule_bins(ps)
    df = pl.DataFrame(
        {
            "apt": apt.astype(str),
            "anull": anull,
            "bin": bins,
            "copy": cls == 1,
            "day": (y > DAYSHIFT_MIN) & (cls != 1),
        }
    ).filter(pl.col("bin") > 0)
    g = df.group_by(["apt", "anull", "bin"]).agg(
        [
            pl.len().alias("n"),
            pl.col("copy").mean().alias("pc"),
            pl.col("day").mean().alias("pd"),
        ]
    )
    table = {}
    for r in g.iter_rows(named=True):
        if r["n"] >= RULE_MIN_N and r["pc"] + r["pd"] >= RULE_MIN_RATE:
            table[(r["apt"], bool(r["anull"]), int(r["bin"]))] = (int(r["n"]), float(r["pc"]), float(r["pd"]))
    return table


def rule_hit(apt, anull, ps, table):
    bins = rule_bins(ps)
    apt = apt.astype(str)
    hit = np.zeros(len(ps), dtype=bool)
    for a, n_, b in table:
        hit |= (apt == a) & (anull == n_) & (bins == b)
    return hit


def apply_rule(pred, reg, apt, anull, ps, table, w, local_pc=None, alpha=0.0):
    out = pred.copy()
    bins = rule_bins(ps)
    apt = apt.astype(str)
    for (a, n_, b), (_, pc, pd) in table.items():
        m = (apt == a) & (anull == n_) & (bins == b)
        if m.any():
            p = np.full(int(m.sum()), pc)
            if local_pc is not None and alpha > 0:
                p = alpha * np.minimum(local_pc[m], 1.0 - pd) + (1.0 - alpha) * pc
            rv = p * ps[m] + pd * (DAY + reg[m]) + (1.0 - p - pd) * reg[m]
            out[m] = w * rv + (1.0 - w) * pred[m]
    return np.clip(out, PRED_MIN, PRED_MAX)


def dataset(x, label, reference=None):
    return lgb.Dataset(
        x,
        label,
        categorical_feature=CAT_IDX,
        feature_name=FEAT_ALL,
        params={"max_bin": 127},
        reference=reference,
        free_raw_data=False,
    )


def local_dataset(x, label, reference=None):
    return lgb.Dataset(
        x,
        label,
        feature_name=LOCAL_FEATS,
        params={"max_bin": 63, "min_data_in_bin": 3},
        reference=reference,
        free_raw_data=False,
    )


def common_params(seed):
    return {
        "verbosity": -1,
        "max_cat_threshold": 64,
        "num_threads": THREADS,
        "max_bin": 127,
        "bagging_freq": 1,
        "feature_fraction": 0.8,
        "seed": seed,
        "bagging_seed": seed + 100,
        "feature_fraction_seed": seed + 200,
    }


def clf_params(seed=0):
    p = common_params(seed)
    p.update(
        {
            "objective": "multiclass",
            "num_class": N_CLASSES,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 300,
            "bagging_fraction": 0.5,
            "lambda_l2": 5.0,
            "cat_smooth": 50,
            "cat_l2": 20,
            "min_data_per_group": 200,
        }
    )
    return p


def reg_params(seed=0):
    p = common_params(seed)
    p.update(
        {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_data_in_leaf": 80,
            "bagging_fraction": 0.7,
            "lambda_l2": 2.0,
            "cat_smooth": 20,
        }
    )
    return p


def spec_params(seed=0):
    p = common_params(seed)
    p.update(
        {
            "objective": "regression",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "lambda_l2": 10.0,
            "cat_smooth": 20,
            "max_cat_threshold": 32,
        }
    )
    return p


def local_params(seed=0):
    return {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_data_in_leaf": 15,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_bin": 63,
        "verbosity": -1,
        "num_threads": THREADS,
        "seed": seed,
        "bagging_seed": seed + 100,
        "feature_fraction_seed": seed + 200,
    }


def mixture(prob, reg, cand):
    vals = np.concatenate([reg[:, None], cand, (reg + DAY)[:, None]], axis=1)
    vals = np.where(np.isnan(vals), reg[:, None], vals)
    return (prob * vals).sum(axis=1)


def spec_mask(cand):
    return np.nan_to_num(cand[:, 0], nan=-1.0) > SPEC_MIN


def blend(mix, spec_pred, mask, w):
    out = mix.copy()
    out[mask] = w * spec_pred + (1 - w) * mix[mask]
    return np.clip(out, PRED_MIN, PRED_MAX)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def logloss(p, yc):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(yc * np.log(p) + (1 - yc) * np.log(1 - p)))


def grid(fn, yv, label):
    print(f"{label} weight -> holdout rmse")
    best_w = 0.0
    best = None
    for w in np.linspace(0.0, 1.0, 11):
        r = rmse(fn(w), yv)
        print(f"  w={w:.1f}: {r:.2f}")
        if best is None or r < best:
            best = r
            best_w = float(w)
    print("chosen weight:", best_w)
    return best_w


def validate(xt, xv, yt, yv, ct, cv, bt, bv, cand_t, cand_v, apt_t, apt_v, an_t, an_v):
    t0 = time.time()

    dtr = dataset(xt, ct)
    dva = dataset(xv, cv, reference=dtr)
    clf = lgb.train(
        clf_params(),
        dtr,
        3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )
    prob = clf.predict(xv, num_iteration=clf.best_iteration)
    print("classifier best iteration:", clf.best_iteration, "time:", round(time.time() - t0), flush=True)
    del dtr, dva

    m0 = ct == 0
    v0 = cv == 0
    rtr = dataset(xt[m0], np.minimum(yt, Y_CLIP)[m0] - bt[m0])
    rva = dataset(xv[v0], yv[v0] - bv[v0], reference=rtr)
    reg = lgb.train(
        reg_params(),
        rtr,
        8000,
        valid_sets=[rva],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)],
    )
    print("regressor best iteration:", reg.best_iteration, "time:", round(time.time() - t0), flush=True)
    del rtr, rva

    rv = bv + reg.predict(xv, num_iteration=reg.best_iteration)
    mix = np.clip(mixture(prob, rv, cand_v), PRED_MIN, PRED_MAX)

    st = spec_mask(cand_t)
    sv = spec_mask(cand_v)
    print("specialist rows: train", int(st.sum()), "holdout", int(sv.sum()), flush=True)
    sdt = dataset(xt[st], yt[st])
    sdv = dataset(xv[sv], yv[sv], reference=sdt)
    spec = lgb.train(
        spec_params(),
        sdt,
        6000,
        valid_sets=[sdv],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(400)],
    )
    print("specialist best iteration:", spec.best_iteration, "time:", round(time.time() - t0), flush=True)
    sp = spec.predict(xv[sv], num_iteration=spec.best_iteration)

    print("")
    spec_w = grid(lambda w: blend(mix, sp, sv, w), yv, "specialist")
    after_spec = blend(mix, sp, sv, spec_w)

    table = rule_table(apt_t, an_t, cand_t[:, 0], ct, yt)
    bins_v = rule_bins(cand_v[:, 0])
    print("")
    print("rule cells: airport, aobt_null, delay bin, n_train, p_copy, p_dayshift, holdout rows")
    for (a, n_, b), (cnt, pc, pd) in sorted(table.items()):
        hits = int(((apt_v.astype(str) == a) & (an_v == n_) & (bins_v == b)).sum())
        print("  ", a, n_, RULE_LABELS[b], cnt, round(pc, 3), round(pd, 3), hits)

    hit_t = rule_hit(apt_t, an_t, cand_t[:, 0], table)
    hit_v = rule_hit(apt_v, an_v, cand_v[:, 0], table)
    print("local copy model rows: train", int(hit_t.sum()), "holdout", int(hit_v.sum()), flush=True)
    yc_t = (ct[hit_t] == 1).astype(np.float64)
    yc_v = (cv[hit_v] == 1).astype(np.float64)
    ldt = local_dataset(xt[hit_t][:, LOCAL_IDX], yc_t)
    ldv = local_dataset(xv[hit_v][:, LOCAL_IDX], yc_v, reference=ldt)
    local = lgb.train(
        local_params(),
        ldt,
        3000,
        valid_sets=[ldv],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)],
    )
    local_iter = max(30, local.best_iteration)
    local_pc = np.full(len(yv), np.nan)
    local_pc[hit_v] = local.predict(xv[hit_v][:, LOCAL_IDX], num_iteration=local_iter)

    bin_pc = np.full(len(yv), np.nan)
    for (a, n_, b), (_, pc, _) in table.items():
        bin_pc[(apt_v.astype(str) == a) & (an_v == n_) & (bins_v == b)] = pc
    print("holdout copy logloss in rule cells: bin rates", round(logloss(bin_pc[hit_v], yc_v), 4),
          "local model", round(logloss(local_pc[hit_v], yc_v), 4))

    print("")
    print("alpha (local model share) -> best rule weight, holdout rmse")
    best = None
    rule_w = 0.0
    alpha = 0.0
    rule_w0 = FALLBACK_RULE_WEIGHT
    for a_ in np.linspace(0.0, 1.0, 11):
        row_best = None
        row_w = 0.0
        for w in np.linspace(0.0, 1.0, 11):
            r = rmse(apply_rule(after_spec, rv, apt_v, an_v, cand_v[:, 0], table, w, local_pc, a_), yv)
            if row_best is None or r < row_best:
                row_best = r
                row_w = float(w)
        print(f"  alpha={a_:.1f}: w={row_w:.1f} rmse={row_best:.2f}")
        if a_ == 0.0:
            rule_w0 = row_w
        if best is None or row_best < best:
            best = row_best
            rule_w = row_w
            alpha = float(a_)
    print("chosen alpha:", alpha, "rule weight:", rule_w, "rule weight without local model:", rule_w0)

    v6_like = apply_rule(after_spec, rv, apt_v, an_v, cand_v[:, 0], table, rule_w0)
    final = apply_rule(after_spec, rv, apt_v, an_v, cand_v[:, 0], table, rule_w, local_pc, alpha)
    np.savez_compressed("holdout_c.npz", y=yv, c6=v6_like, c7=final)
    hit = hit_v

    oracle = mixture(np.eye(N_CLASSES)[cv], rv, cand_v)
    err = (final - yv) ** 2
    total = err.sum()

    print("")
    print("holdout (Jan+Jul, unfiltered), rows:", len(yv))
    print("  proxy base:          ", round(rmse(bv, yv), 2))
    print("  regression only:     ", round(rmse(rv, yv), 2))
    print("  soft mixture:        ", round(rmse(mix, yv), 2))
    print("  mixture + specialist:", round(rmse(after_spec, yv), 2))
    print("  + rule, no local:    ", round(rmse(v6_like, yv), 2))
    print("  + rule:              ", round(rmse(final, yv), 2))
    print("  oracle class:        ", round(rmse(oracle, yv), 2))
    print("  rows hit by rule:", int(hit.sum()), "rmse on them: before", round(rmse(after_spec[hit], yv[hit]), 1),
          "v6 way", round(rmse(v6_like[hit], yv[hit]), 1), "now", round(rmse(final[hit], yv[hit]), 1))
    print("  share of squared error from y>43200:", round(float(err[yv > DAYSHIFT_MIN].sum() / total), 3))

    print("")
    print("per class (true): n, mean prob of true class, rmse final, share of squared error")
    for k in range(N_CLASSES):
        mask = cv == k
        if mask.sum():
            print(
                "  ",
                CLASS_NAMES[k],
                int(mask.sum()),
                round(float(prob[mask, k].mean()), 3),
                round(rmse(final[mask], yv[mask]), 1),
                round(float(err[mask].sum() / total), 3),
            )

    print("")
    print("per airport: n, mixture, final, share of squared error")
    for a in np.unique(apt_v):
        mask = apt_v == a
        print(
            "  ",
            a,
            int(mask.sum()),
            round(rmse(mix[mask], yv[mask]), 1),
            round(rmse(final[mask], yv[mask]), 1),
            round(float(err[mask].sum() / total), 3),
        )

    worst = np.argsort(err)[::-1][:15]
    print("")
    print("worst 15 holdout rows: airport, true, final, v6 way, class, p_sched, aobt_null, local copy prob")
    for i in worst:
        print(
            "  ",
            apt_v[i],
            int(yv[i]),
            int(final[i]),
            int(v6_like[i]),
            CLASS_NAMES[cv[i]],
            None if np.isnan(cand_v[i, 0]) else int(cand_v[i, 0]),
            bool(an_v[i]),
            None if np.isnan(local_pc[i]) else round(float(local_pc[i]), 3),
        )

    importance = sorted(
        zip(local.feature_name(), local.feature_importance("gain")), key=lambda kv: kv[1], reverse=True
    )
    print("")
    print("top local model features:")
    for n, g in importance[:10]:
        print(f"  {n}: {g:,.1f}")
    print("", flush=True)

    return clf.best_iteration, reg.best_iteration, spec.best_iteration, local_iter, spec_w, rule_w, alpha, rule_w0


def key_frame(frame, cls=None):
    k = frame.select(KEY_COLS)
    if cls is not None:
        k = k.with_columns(
            [
                pl.Series("t_sched", (cls == 1).astype(np.float64)),
                pl.Series("t_copy", ((cls >= 1) & (cls <= 4)).astype(np.float64)),
            ]
        )
    return k


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(DATA, "training_*.parquet")))
    if not files:
        sys.exit("no training parquet files found")

    print("loading training", flush=True)
    full = pl.concat([read_fixed(f) for f in files], how="vertical_relaxed")
    print("rows:", full.height, flush=True)
    print("building training features", flush=True)
    train = add_labels(build(full))
    del full
    train = train.filter(pl.col(TARGET).is_not_null())
    print("departures:", train.height, "time:", round(time.time() - t0), flush=True)
    print(train.group_by("cls").len().sort("cls"))

    maps = category_maps(train)
    x = matrix(train, maps)
    y = train[TARGET].cast(pl.Float64).to_numpy()
    cls = train["cls"].to_numpy().astype(np.int32)
    base = train["base"].cast(pl.Float64).to_numpy()
    cand = candidates(train)
    apt = train["ADEP_mvt"].to_numpy().astype(str)
    anull = train["aobt_null"].to_numpy() > 0
    month = train["month"].to_numpy()
    keyf = key_frame(train, cls)
    del train

    if SKIP_VALID:
        clf_rounds, reg_rounds, spec_rounds, local_rounds, spec_w, rule_w, alpha, rule_w0 = (
            FALLBACK_CLF_ROUNDS,
            FALLBACK_REG_ROUNDS,
            FALLBACK_SPEC_ROUNDS,
            FALLBACK_LOCAL_ROUNDS,
            FALLBACK_SPEC_WEIGHT,
            FALLBACK_RULE_WEIGHT,
            FALLBACK_ALPHA,
            FALLBACK_RULE_WEIGHT,
        )
    else:
        va = np.isin(month, VALID_MONTHS)
        tr = ~va
        k_tr = keyf.filter(pl.Series(tr))
        k_va = keyf.filter(pl.Series(va))
        print("building target encodings for holdout", flush=True)
        xt = np.hstack([x[tr], te_oof(k_tr)])
        xv = np.hstack([x[va], te_full(k_tr, k_va)])
        clf_rounds, reg_rounds, spec_rounds, local_rounds, spec_w, rule_w, alpha, rule_w0 = validate(
            xt, xv, y[tr], y[va], cls[tr], cls[va], base[tr], base[va], cand[tr], cand[va],
            apt[tr], apt[va], anull[tr], anull[va],
        )
        del xt, xv, k_tr, k_va

    clf_rounds = max(50, int(clf_rounds * REFIT_SCALE))
    reg_rounds = max(100, int(reg_rounds * REFIT_SCALE))
    spec_rounds = max(50, int(spec_rounds * REFIT_SCALE))
    local_rounds = max(30, int(local_rounds * REFIT_SCALE))
    print(
        "refit rounds: classifier", clf_rounds, "regressor", reg_rounds, "specialist", spec_rounds,
        "local", local_rounds, "spec weight", spec_w, "rule weight", rule_w, "alpha", alpha, flush=True,
    )

    table_all = rule_table(apt, anull, cand[:, 0], cls, y)
    hit_a = rule_hit(apt, anull, cand[:, 0], table_all)

    xa = np.hstack([x, te_oof(keyf)])
    del x

    clf = lgb.train(clf_params(), dataset(xa, cls), clf_rounds)
    print("classifier refit done, time:", round(time.time() - t0), flush=True)

    m0 = cls == 0
    reg_set = dataset(xa[m0], np.minimum(y, Y_CLIP)[m0] - base[m0])
    regs = []
    for seed in SEEDS:
        regs.append(lgb.train(reg_params(seed), reg_set, reg_rounds))
        print("regressor seed", seed, "done, time:", round(time.time() - t0), flush=True)
    del reg_set

    sa = spec_mask(cand)
    spec_set = dataset(xa[sa], y[sa])
    specs = [lgb.train(spec_params(seed), spec_set, spec_rounds) for seed in SEEDS]
    print("specialist done, time:", round(time.time() - t0), flush=True)

    local_set = local_dataset(xa[hit_a][:, LOCAL_IDX], (cls[hit_a] == 1).astype(np.float64))
    locals_ = [lgb.train(local_params(seed), local_set, local_rounds) for seed in SEEDS]
    print("local model done, rows:", int(hit_a.sum()), "time:", round(time.time() - t0), flush=True)
    del xa, spec_set, local_set

    def predict_file(rank_file, sub_file, tag):
        print("loading", rank_file, flush=True)
        rank = read_fixed(os.path.join(DATA, rank_file))
        sub = pl.read_parquet(os.path.join(DATA, sub_file))
        rdep = build(rank)
        del rank
        print("departures:", rdep.height, "template rows:", sub.height, flush=True)

        xr = np.hstack([matrix(rdep, maps), te_full(keyf, key_frame(rdep))])
        base_r = rdep["base"].cast(pl.Float64).to_numpy()
        cand_r = candidates(rdep)
        apt_r = rdep["ADEP_mvt"].to_numpy().astype(str)
        anull_r = rdep["aobt_null"].to_numpy() > 0
        prob = clf.predict(xr)
        reg_r = base_r + np.mean([m.predict(xr) for m in regs], axis=0)
        mix = np.clip(mixture(prob, reg_r, cand_r), PRED_MIN, PRED_MAX)
        sr = spec_mask(cand_r)
        if sr.sum():
            sp = np.mean([m.predict(xr[sr]) for m in specs], axis=0)
            pred = blend(mix, sp, sr, spec_w)
        else:
            pred = mix

        hit_r = rule_hit(apt_r, anull_r, cand_r[:, 0], table_all)
        local_pc_r = np.full(len(pred), np.nan)
        if hit_r.sum():
            local_pc_r[hit_r] = np.mean([m.predict(xr[hit_r][:, LOCAL_IDX]) for m in locals_], axis=0)
        pred_a6 = apply_rule(pred, reg_r, apt_r, anull_r, cand_r[:, 0], table_all, rule_w0)
        pred_a7 = apply_rule(pred, reg_r, apt_r, anull_r, cand_r[:, 0], table_all, rule_w, local_pc_r, alpha)
        print("rows in specialist region:", int(sr.sum()), "rows hit by rule:", int(hit_r.sum()))
        print("months:", sorted(rdep["MVT_TIME_UTC_mvt"].dt.month().unique().to_list()))

        ids = rdep["MVT_ID_mvt"].cast(pl.Float64)
        for variant, values in (("c6", pred_a6), ("c7", pred_a7)):
            out = sub.select("MVT_ID_mvt").join(
                pl.DataFrame({"MVT_ID_mvt": ids, "pred": values.astype(np.float64)}), on="MVT_ID_mvt", how="left"
            )
            assert out.height == sub.height, (out.height, sub.height)
            print(variant, tag, "template rows without a prediction (filled with 900):", out["pred"].null_count())
            out = out.with_columns(pl.col("pred").fill_null(900.0))
            name = f"pred_{variant}_{tag}.parquet"
            out.write_parquet(name)
            print("written:", name, out.shape, "mean", round(float(out["pred"].mean()), 1), "max", round(float(out["pred"].max()), 0))
        print("time:", round(time.time() - t0), flush=True)

    print("rule cells used:")
    for (a, n_, b), (cnt, pc, pd) in sorted(table_all.items()):
        print("  ", a, n_, RULE_LABELS[b], cnt, round(pc, 3), round(pd, 3))

    predict_file("ranking.parquet", "submitting.parquet", "ranking")
    if os.path.exists(os.path.join(DATA, "final_ranking.parquet")) and os.path.exists(
        os.path.join(DATA, "final_submitting.parquet")
    ):
        predict_file("final_ranking.parquet", "final_submitting.parquet", "final")
    else:
        print("final_ranking.parquet or final_submitting.parquet not found, skipping final predictions")

if __name__ == "__main__":
    main()
