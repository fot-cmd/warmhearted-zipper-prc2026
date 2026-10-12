#!/usr/bin/env bash
set -euo pipefail

export DC_DATA="${DC_DATA:-.}"
export DC_SEEDS="${DC_SEEDS:-0,1,2}"
FINAL_VARIANTS="${FINAL_VARIANTS:-c6,c7}"
RUN_D="${RUN_D:-0}"
PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p logs
"$PY" "$HERE/model_c.py" 2>&1 | tee logs/model_c.log
if [ "$RUN_D" = "1" ]; then
    "$PY" "$HERE/model_d.py" 2>&1 | tee logs/model_d.log
fi
DC_VARIANTS="$FINAL_VARIANTS" DC_VERSION=11 DC_WRITE_FINAL=1 "$PY" "$HERE/ensemble.py" 2>&1 | tee logs/ensemble.log
