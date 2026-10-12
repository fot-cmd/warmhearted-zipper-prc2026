# PRC Data Challenge 2026 - team `warmhearted-zipper`

Taxi-out time prediction (`TAXITIME_SEC_mvt`) for departures at ten European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM).

Licensed under GPLv3 (see `LICENSE`). No external data is used: only the files
distributed by the challenge (`training_*.parquet`, `ranking.parquet`,
`submitting.parquet`, `final_ranking.parquet`, `final_submitting.parquet`).

## The key observation

The target is `MVT_TIME - BLOCK_TIME`. For a large share of departures the block-off
time is **not a real measurement**: it is a copy (within a few seconds of jitter) of
another timestamp that is available in the data - the scheduled time, `AOBT_3`,
`EOBT_1`, `LOBT`/`IOBT`. When that happens the "taxi time" is really the departure
delay, which can be hours. A further small class is shifted by one day
(`y ≈ 86400 + taxi`).

These copied rows dominate the RMSE. A model that knew, for every row, which source
the block time came from (an "oracle class") would score roughly at the level of the
leaderboard leader. Almost all of the remaining error is therefore about
**identifying copied rows**, not about modelling real taxi physics.

## Model

Each model is a mixture of experts, trained per run on all airports together:

1. **Source classifier** (LightGBM multiclass, 6 classes: `model`, `sched`, `aobt`,
   `eobt`, `lobt_iobt`, `dayshift`). It predicts where the block time came from.
2. **Residual regressor** for the `model` class: target clipped at 3 h, baseline
   = clipped `MVT - AOBT`.
3. **Expected value**:
   `E[y] = p_model·reg + Σ_k p_k·candidate_k + p_dayshift·(reg + 86400)`,
   where `candidate_k = MVT - source_k`.
4. **Specialist regressor** for delayed rows (`MVT - SCHED > 1 h`), blended in.
5. **Rule cells** (airport × AOBT missing × delay bin) using empirical copy and
   day-shift rates from training; most important at LIRF. Optionally mixed with a
   small local LightGBM copy model (weight `alpha`).

### Features
- Timestamp differences between movement time and every candidate source,
  rounding/second patterns, hour/day features.
- Period context per airport: arrival copy rates over the day / last 3 h,
  departure counts, share of departures without AOBT, median delay.
- Out-of-fold target encodings of flight number, stand and callsign prefix.
- Callsign shape (length, leading letters, digit count, ends in a letter), missing
  aircraft type, local-time and UTC day crossing between schedule and movement
  (model D).
- **Inbound stand features (models C and D)** - the most useful addition late in the
  competition. Each departure is linked (`join_asof`, backward, same stand,
  ≤ 24 h) to the previous arrival at the same stand. Arrival block times are
  visible in the ranking and final data, so we know whether the inbound block-in
  was itself a copy of its schedule (`in_copy`), plus the gap, the inbound delay and
  a two-arrival copy average. Copying behaviour turns out to be strongly correlated
  along a stand's turnaround chain.

### Variants and ensemble
| variant | file | description |
|---|---|---|
| c6 | `model_c.py` | v7-style parameters, rule cells only |
| c7 | `model_c.py` | as c6, rule cells mixed with the local copy model |
| d8 | `model_d.py` | large regressor (1023 leaves), new features, per-segment specialist weights, day-crossing rule |
| d9 | `model_d.py` | as d8 with a single global specialist weight and the plain rule |

`ensemble.py` takes an equal-weight average of the chosen variants
(`DC_VARIANTS`). Weights were deliberately **not** tuned on the leaderboard.

**Final submission: equal-weight average of c6 and c7** (leaderboard 312.02, our best).
Adding d8 and d9 scored 312.66, so model D is kept in the repository as a documented
experiment but is not part of the final file.

## Validation lessons

- Holdout = January and July 2025 (the months matching the ranking period).
- The holdout RMSE is dominated by a handful of rows (one flight with a huge
  day-shifted value, two cancelled flights at LFPG). Several changes that improved
  the holdout made the leaderboard worse (v8, v9). We therefore made design choices
  a priori and used the leaderboard only to accept or reject a whole model.
- For blends we used the exact identity
  `MSE(mean) = mean_i MSE_i - (1/n²) Σ_{i<j} D_ij`,
  where `D_ij` is the mean squared difference between two submissions. Blend scores
  are computed from known scores without spending uploads (`analysis/` has the
  diagnostic scripts).

## Leaderboard history (ranking RMSE)

| version | content | RMSE |
|---|---|---|
| v2 | single LightGBM regressor | 533.40 |
| v5 | source classifier + mixture + delay specialist | 355.69 |
| v6 | + rule cells (airport × AOBT missing × delay bin) | 315.06 |
| v7 | + local copy model | 314.90 |
| v8 | big regressor, new features (holdout gain did not transfer) | 317.55 |
| v9 | v8 variant | 318.52 |
| v10 | ensemble a6, a7, b8, b9 | 314.48 |
| v11 | inbound-stand features, ensemble c6, c7 | 312.02 |
| v12 | ensemble c6, c7, d8, d9 | 312.66 |

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
DC_DATA=/path/to/challenge/data ./run_all.sh
```

Outputs are written to the current directory:
`warmhearted-zipper_v11.parquet` (ranking) and `warmhearted-zipper_final.parquet`
(final). Options (environment variables):

| variable | default | meaning |
|---|---|---|
| `DC_DATA` | `.` | folder with the challenge parquet files |
| `DC_SEEDS` | `0,1,2` | seeds averaged in every LightGBM stage |
| `DC_THREADS` | `0` (all) | LightGBM threads |
| `DC_SKIP_VALID` | `0` | `1` skips the holdout run (uses fallback round counts) |
| `FINAL_VARIANTS` | `c6,c7` | variants averaged into the final file |
| `RUN_D` | `0` | `1` also trains model D (variants d8, d9) |

Each model script can also be run on its own; `ensemble.py` then combines the
`pred_<variant>_<tag>.parquet` files.

Hardware used: 8-core desktop, 24 GB RAM, Fedora 43. Runtime is roughly 40 minutes
per model script with three seeds. Peak memory stays below 20 GB.

## Files

- `docs/CODE.md` - detailed code documentation: every function, the class labels, and the design decisions.

- `features_base.py` - data loading and generic traffic/context features.
- `model_c.py`, `model_d.py` - the two model families described above.
- `ensemble.py` - equal-weight averaging and writing the submission files.
- `run_all.sh` - full pipeline (model C and the final ensemble; `RUN_D=1` adds model D).
- `analysis/` - exploratory scripts used to find the copy mechanism
  (`diag.py`, `diag2.py`), compare 2025 vs 2026 behaviour (`diag3.py`) and measure
  the inbound-stand effect (`diag5.py`). Run them from the data folder.
