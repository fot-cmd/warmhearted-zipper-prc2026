# Code documentation

This document explains how the code works, file by file, and why each design
decision was made. The final submission is produced by `model_c.py` (variants c6
and c7) and `ensemble.py`. `model_d.py` is a documented experiment.

## 1. Pipeline overview

```
training_*.parquet ──► build() ──► add_labels() ──► holdout validation (Jan+Jul 2025)
                                                     │  picks rounds and blend weights
                                                     ▼
                                   refit on all of 2025 (3 seeds)
                                                     │
ranking.parquet / final_ranking.parquet ──► build() ─┴► predict ──► pred_<variant>_<tag>.parquet
                                                                          │
                                                          ensemble.py ◄───┘
                                                              │
                                     warmhearted-zipper_v11.parquet / _final.parquet
```

Everything is computed per airport (`build()` loops over the ten departure
airports), because copying behaviour and traffic patterns are airport specific.

## 2. What information is used at prediction time

The ranking and final files hide the departure block-off time (that is the
target). Everything else in the file is available. Our features use:

- the departure row itself: movement time, schedule, `AOBT_3`, `EOBT_1`, `LOBT`,
  `IOBT`, `ARVT`, stand, runway, flight number, aircraft type, operator;
- other movements at the same airport in the same file: departures (without
  their block time) and arrivals (including arrival block-in time and arrival
  taxi time, which are visible in the files).

Labels (copy class, taxi time) are taken only from the 2025 training files.
Target encodings are fitted on 2025 only and then applied to 2026 rows. No
external data is used.

## 3. `features_base.py`

Generic traffic and timestamp features, shared by both models
(`prepare(frame, full)` is the only entry point used by the models).

| function | purpose |
|---|---|
| `load_training` | reads all `training_*.parquet` files from `DC_DATA` |
| `cumulative_events` | sorted event table with a running count (and optionally a running sum) per key; turns window counts into two as-of lookups |
| `asof_lookup` | backward `join_asof` of anchor times onto an event table |
| `window_aggregate` | number of events (or mean value) in a backward or forward time window, computed as the difference of two cumulative counts |
| `active_count` | number of open intervals at a given time (started minus finished) |
| `departure_spans` / `arrival_spans` | time intervals during which an aircraft is taxiing out / in |
| `add_gaps` | seconds to the previous / next departure at the airport and on the same runway |
| `add_context` | traffic features: departures, arrivals and runway movements in 15/30/60 min windows back and 15/30 min forward, mean arrival taxi time, aircraft currently taxiing |
| `add_features` | proxies `p_* = movement time - candidate off-block time` for AOBT, LOBT, EOBT, IOBT and schedule; differences between those timestamps; cyclic time features; `base` = first plausible proxy (60 s to 1 h), used as the regression baseline |
| `prepare` | `add_context` followed by `add_features` |

Design decision: window counts are computed with cumulative counts and as-of
joins instead of rolling windows, so the cost is linear in the number of
movements and memory stays low on a 24 GB machine.

## 4. `model_c.py`

### 4.1 Reading and features

| function | purpose |
|---|---|
| `read_fixed` | reads a parquet file and converts a non-standard Arrow extension type in `AIRCRAFT_OPERATOR_flt` to plain strings |
| `add_extra` | seconds of the movement timestamp, missing AOBT / EOBT flags, schedule minute modulo 5, EOBT and LOBT relative to schedule, flight time from ARVT, callsign prefix |
| `period_key`, `period_tables`, `add_period_context` | airport context per calendar day and per 3-hour block: share of arrivals whose block-in equals their schedule (`arr_copy_*`), arrival taxi statistics, number of departures, share of departures without AOBT, median delay. Copying is a property of how an airport reports data on a given day, so these columns tell the model when copying is likely |
| `add_inbound` | links each departure to the previous arrival at the same stand (backward as-of join on time, at most 24 h). Columns: `in_copy` (was the inbound block-in a copy of its schedule), `in_gap_h`, `in_delay`, `in_copy_prev2` (average over the last two inbound arrivals). Copying turned out to be correlated along a stand's turnaround chain |
| `build` | runs all feature steps per airport and concatenates the result |

### 4.2 Labels

`add_labels` assigns every training departure to one of six classes, in this
order of priority:

| class | meaning | condition |
|---|---|---|
| 1 `sched` | block time copied from the schedule | `abs(BLOCK - SCHED) <= 6 s` |
| 2 `aobt` | copied from `AOBT_3` | `abs(BLOCK - AOBT_3) <= 6 s` |
| 3 `eobt` | copied from `EOBT_1` | `abs(BLOCK - EOBT_1) <= 6 s` |
| 4 `lobt_iobt` | copied from `LOBT` or `IOBT` | `<= 6 s` from either |
| 5 `dayshift` | block time on the previous day | taxi time > 12 h and none of the above |
| 0 `model` | a real measurement | everything else |

`candidates` returns, for each row, the taxi time each copy class would imply
(`p_sched`, `p_aobt`, `p_eobt`, `p_lobt` or `p_iobt`).

### 4.3 Target encodings

`te_stats`, `te_apply`, `te_oof`, `te_full`: smoothed means of "copied from
schedule" and "copied from any source" per flight number, per stand and per
callsign prefix. Inside the training set they are out-of-fold (5 folds) to avoid
leakage; for ranking/final rows they are fitted on all of 2025.

### 4.4 Models

| model | params function | target | rows |
|---|---|---|---|
| source classifier | `clf_params` | class 0-5 | all departures |
| residual regressor | `reg_params` | `min(y, 3 h) - base` | class 0 only |
| delay specialist | `spec_params` | `y` | rows with `p_sched > 1 h` |
| local copy model | `local_params` | copied from schedule (0/1) | rows inside a rule cell |

All are LightGBM. `dataset`/`local_dataset` build the LightGBM datasets,
`common_params` holds shared settings.

### 4.5 Combining the models

- `mixture`: expected value over the classes,
  `p0·reg + Σ pk·candidate_k + p5·(reg + 86400)`. A missing candidate falls back
  to `reg`.
- `spec_mask`, `blend`: in the delayed region the mixture is blended with the
  specialist with weight `w`.
- `rule_bins`, `rule_table`, `rule_hit`, `apply_rule`: rule cells
  (airport × AOBT missing × delay bin). For each cell with at least 5 training
  rows and a copy + day-shift rate of at least 20 %, the expected value
  `p·p_sched + pd·(86400 + reg) + (1 - p - pd)·reg` replaces part of the
  prediction. `p` is the empirical copy rate of the cell, optionally mixed with
  the local copy model (`alpha`). In practice all cells are at LIRF.

Variants: **c6** applies the rule with the empirical rate only; **c7** mixes in
the local copy model.

### 4.6 Validation and refit

`validate` trains every stage on Feb-Jun and Aug-Dec 2025 and evaluates on
January and July 2025 (the months of the ranking period). It chooses the number
of boosting rounds (early stopping), the specialist weight, the rule weight and
`alpha` on 11-point grids, and prints diagnostics (per class, per airport, worst
rows, oracle score). `main` then refits on all of 2025 with 1.1 × the chosen
rounds and three seeds, and `predict_file` writes predictions for `ranking` and
`final`. `rmse`, `logloss`, `grid` and `key_frame` are small helpers.

Design decisions:

- **Mixture instead of a single regressor.** A regressor trained on all rows
  averages a 15-minute taxi and a 5-hour copied delay into a useless middle
  value. Predicting where the block time came from and taking the expected
  value is far better (holdout RMSE roughly 560 → 410).
- **Residual regression on a clipped target.** Real taxi times are short; the
  regressor learns `y - base` on class 0 rows only, so copies and day shifts do
  not distort it.
- **Specialist for delayed flights.** Copy rows dominate the squared error and
  live almost entirely in flights that took off more than 1 h after schedule.
- **Rule cells only where the data supports them.** Minimum row count and
  minimum rate keep the rule from firing on noise.
- **No leaderboard-tuned weights.** All weights are chosen on the 2025 holdout;
  the leaderboard was used only to accept or reject complete models.

## 5. `model_d.py` (experiment, not in the final file)

Same pipeline as model C with these changes:

- `new_features`: callsign length, leading letters, digit count, ending in a
  letter, missing aircraft type, day change between schedule and movement in
  local time and in UTC.
- Larger regressor (`num_leaves` 1023).
- `spec_segments`, `blend_segments`: separate specialist weight per delay bin ×
  AOBT missing segment.
- `rule_table_dc`, `apply_rule_dc`: rule cells further split by day change,
  shrunk towards the parent cell (`RULE_SHRINK`).

Variants d8 (segments + day-change rule) and d9 (global weight + plain rule).
These improved the holdout (321 vs 332) but the four-model ensemble scored
312.66 on the leaderboard against 312.02 for model C alone, so it was left out.

## 6. `ensemble.py`

Reads `pred_<variant>_<tag>.parquet` for the variants in `DC_VARIANTS`, checks
that every template row has a prediction, averages them with equal weights,
rounds to integer seconds and writes a file with exactly the template's schema.
`DC_WRITE_FINAL=1` also writes `warmhearted-zipper_final.parquet`.
