#!/usr/bin/env bash
# Stage 2: piecewise-constant baseline on limited data (Table 1).
# Arguments are iteration budgets (default: "10000 80000"):
#   10000 - equal budget with the CLR runs, ~16 min per run, ~1 h total
#   80000 - the paper's protocol, ~2 h per run, ~8.5 h total
#
#   bash scripts/stage2_subsets_pc.sh 10000 2>&1 | tee -a results/stage2_subsets_pc.log
#   bash scripts/stage2_subsets_pc.sh 80000 2>&1 | tee -a results/stage2_subsets_pc.log

cd "$(dirname "$0")/.."
budgets=("$@")
[ ${#budgets[@]} -eq 0 ] && budgets=(10000 80000)

while pgrep -f "python3? -m hw_2.train" >/dev/null; do
    echo "waiting for a running hw_2.train to finish..."
    sleep 60
done

run() {
    echo ">>> $(date '+%F %T') $*"
    uv run python -m hw_2.train "$@" || echo "!!! skipped or failed: $*"
}

for iters in "${budgets[@]}"; do
    for n in 10000 20000 30000 40000; do        # paper, PC 80k: 71.4 / 82.7 / 85.7 / 89.1%
        run --schedule piecewise --lr 0.35 --warmup-iters 200 --iters "$iters" --train-subset $n
    done
done
