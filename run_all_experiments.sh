#!/bin/bash
# ======================================================
# Orchestration script — runs all 4 techniques in parallel
# per configuration, then moves to the next config
# ======================================================

set -e

run_config() {
    local du=$1
    local ru=$2
    local n_ue=$3
    local label=$4

    echo "  Launching all 4 techniques: du=$du ru=$ru n_ue=$n_ue -> *_${label}.txt"

    python DUSA-UL-Corrected.py   --du_max "$du" --ru_max "$ru" --n_ue "$n_ue" > "ul_dusa_${label}.txt"           2>&1 &
    python TRAD-UL-Corrected.py   --du_max "$du" --ru_max "$ru" --n_ue "$n_ue" > "ul_trad_${label}.txt"           2>&1 &
    python STATIC-UL-Corrected.py --du_max "$du" --ru_max "$ru" --n_ue "$n_ue" > "ul_sem5qi-static_${label}.txt"  2>&1 &
    python RAND-UL-Corrected.py   --du_max "$du" --ru_max "$ru" --n_ue "$n_ue" > "ul_dynamic-random_${label}.txt" 2>&1 &

    wait  # block until all 4 finish before next config
    echo "  Done: $label"
}

# ======================================================
# Experiment 1: Compute budget sweep (3 UEs per path = 12)
# ======================================================
echo "====== Generating shared inputs ======"
python generate_shared_inputs.py
echo "====== EXPERIMENT 1: Compute Budget Sweep ======"
run_config 2.0 0.5 12 1cb
run_config 3.5 1.0 12 2cb
run_config 5.0 2.0 12 3cb
run_config 7.0 3.5 12 4cb

# ======================================================
# Experiment 2: UE count sweep (CB3 fixed)
# ======================================================
echo "====== EXPERIMENT 2: UE Count Sweep ======"
run_config 5.0 2.0  4  1ue
run_config 5.0 2.0 12  3ue
run_config 5.0 2.0 20  5ue
run_config 5.0 2.0 40 10ue

echo "====== ALL EXPERIMENTS COMPLETE ======"
