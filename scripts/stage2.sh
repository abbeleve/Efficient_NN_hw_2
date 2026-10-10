#!/usr/bin/env bash
# Stage 2: reproduction runs for ResNet-56 / CIFAR-10, in priority order.
# Runs sequentially (one GPU); finished runs (results/<name>/summary.json) are skipped,
# so the script can be interrupted and restarted at any time.
#
#   bash scripts/stage2.sh 2>&1 | tee -a results/stage2.log
#
# Times assume ~0.1 s/iteration at batch 1000 on an RTX 5060 Ti.

cd "$(dirname "$0")/.."

# Do not compete for the GPU with runs started by hand.
while pgrep -f "python3? -m hw_2.train" >/dev/null; do
    echo "waiting for a running hw_2.train to finish..."
    sleep 60
done

run() {
    echo ">>> $(date '+%F %T') $*"
    uv run python -m hw_2.train "$@" || echo "!!! skipped or failed: $*"
}

CLR=(--schedule onecycle --lr 0.1 --max-lr 3)
PC=(--schedule piecewise --lr 0.35 --warmup-iters 200)

# --- 1. Equal-budget control and cycle length (Fig. 1a/1b, Table 6), ~2.3 h -------------
run "${PC[@]}" --iters 10000                    # PC with the same budget as CLR-10k
run "${CLR[@]}" --iters 20000                   # paper: 92.7%
run "${PC[@]}" --iters 20000
for it in 4000 6000 8000; do                    # paper: 91.1 / 92.1 / 91.7%
    run "${CLR[@]}" --iters $it
done

# --- 2. Weight decay: balance of regularization (Fig. 12b), ~50 min ------------------
for wd in 1e-3 1e-5 1e-6; do                    # 1e-4 is the main run
    run "${CLR[@]}" --iters 10000 --wd $wd
done

# --- 3. Seeds for the main comparison, ~1.1 h ------------------------------------------
for seed in 1 2; do
    run "${CLR[@]}" --iters 10000 --seed $seed
    run "${PC[@]}" --iters 10000 --seed $seed
done

# --- 4. Peak learning rate (Table 6), ~1.4 h ---------------------------------------------
for max_lr in 1 1.5 2 2.5 3.5; do               # paper: 91.3 / 90.9 / 91.7 / 92.3 / 92.1%
    run --schedule onecycle --lr 0.1 --max-lr $max_lr --iters 10000
done

# --- 5. Limited training data with CLR (Table 1), ~1.1 h ---------------------------------
for n in 10000 20000 30000 40000; do            # paper: 80.6 / 87.9 / 89.6 / 91.1%
    run "${CLR[@]}" --iters 10000 --train-subset $n
done

# --- 6. Well-tuned standard baseline (He et al. 2016: 93.03%), ~0.5 h ------------------
run --schedule piecewise --lr 0.1 --batch-size 128 --iters 64000 --milestones 32000 48000 \
    --eval-every 1000

# --- 7. LR range test length independence (Fig. 2b), ~35 min -----------------------------
run --schedule range --lr 0 --max-lr 3 --iters 20000 --eval-every 400
