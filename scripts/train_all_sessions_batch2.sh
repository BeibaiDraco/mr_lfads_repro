#!/usr/bin/env bash
#
# Train MR-LFADS on 9 real-data sessions sequentially (batch 2).
# Designed to survive terminal disconnection (run via nohup).
#
# Usage:
#   nohup bash scripts/train_all_sessions_batch2.sh > logs/train_batch2.log 2>&1 &
#
set -euo pipefail

cd /home/mr_lfads_repro

SESSIONS=("20201004" "20201204" "20201211" "20201216" "20231103" "20231109" "20231121" "20231123" "20231130")
CONFIG="configs/rct_fef_lip_sc.json"

mkdir -p logs

echo "========================================"
echo "Starting batch 2 training: ${#SESSIONS[@]} sessions"
echo "Time: $(date)"
echo "========================================"

for DATE in "${SESSIONS[@]}"; do
    echo ""
    echo "========================================"
    echo "SESSION: ${DATE}"
    echo "Start: $(date)"
    echo "========================================"

    DATA_FILE="data/rct_${DATE}.npz"
    RUN_DIR="runs/rct_${DATE}"

    # Step 1: Prepare combined dataset if not already done
    if [ ! -f "${DATA_FILE}" ]; then
        echo "[${DATE}] Preparing dataset..."
        python scripts/prepare_session_data.py --date "${DATE}"
    else
        echo "[${DATE}] Dataset already exists: ${DATA_FILE}"
    fi

    # Step 2: Check trial count and adjust batch size for small sessions
    N_TRIALS=$(python -c "import numpy as np; d=np.load('${DATA_FILE}'); print(d['region0'].shape[0])")
    echo "[${DATE}] Trials: ${N_TRIALS}"

    BATCH_SIZE=64
    if [ "${N_TRIALS}" -lt 120 ]; then
        BATCH_SIZE=32
        echo "[${DATE}] Small session — using batch_size=${BATCH_SIZE}"
    fi

    # Step 3: Create a session-specific config with correct batch_size
    SESSION_CONFIG="configs/rct_${DATE}.json"
    python -c "
import json
with open('${CONFIG}') as f:
    cfg = json.load(f)
cfg['batch_size'] = ${BATCH_SIZE}
with open('${SESSION_CONFIG}', 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'Config written: ${SESSION_CONFIG}  (batch_size={cfg[\"batch_size\"]})')
"

    # Step 4: Train (|| true to not abort the whole script if one session fails)
    echo "[${DATE}] Training..."
    if python scripts/train_mr_lfads.py \
        --data "${DATA_FILE}" \
        --save_dir "${RUN_DIR}" \
        --config_json "${SESSION_CONFIG}"; then
        echo "[${DATE}] Training complete: $(date)"

        # Step 5: Visualize
        echo "[${DATE}] Generating figures..."
        python scripts/visualize_results.py \
            --run_dir "${RUN_DIR}" \
            --data "${DATA_FILE}" || echo "[${DATE}] Visualization failed, continuing..."
    else
        echo "[${DATE}] Training FAILED — continuing to next session: $(date)"
    fi

    echo "[${DATE}] Done: $(date)"
    echo "========================================"
done

echo ""
echo "========================================"
echo "ALL SESSIONS COMPLETE: $(date)"
echo "========================================"
