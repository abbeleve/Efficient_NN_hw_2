"""Builds all report figures and prints the result tables from results/.

    uv run python scripts/make_figures.py

Figures go to report/figures/; runs that do not exist yet are skipped.
"""

import json
import statistics
from pathlib import Path

from hw_2.plots import plot_curves, plot_range, plot_schedule, plot_sweep

RES = Path("results")
FIG = Path("report/figures")
B = "cifar10_resnet56_"


def run(name: str) -> Path:
    return RES / (B + name)


def final_acc(name: str) -> float | None:
    path = run(name) / "summary.json"
    if not path.exists():
        return None
    acc = json.loads(path.read_text())["final_test_acc"]
    return None if acc is None else acc * 100


def existing(*names: str) -> list[Path]:
    return [run(n) for n in names if (run(n) / "summary.json").exists()]


def clr(it, wd="0.0001", seed=0, max_lr="3", n=None):
    return f"onecycle_lr0.1-{max_lr}_it{it}_bs1000_wd{wd}" + (f"_n{n}" if n else "") + f"_s{seed}"


def pc(it, seed=0, n=None):
    return f"piecewise_lr0.35_it{it}_bs1000_wd0.0001_wu200" + (f"_n{n}" if n else "") + f"_s{seed}"


HE = "piecewise_lr0.1_it64000_bs128_wd0.0001_s0"


def mean_std(names):
    vals = [v for v in (final_acc(n) for n in names) if v is not None]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None), len(vals)


def curves(names, labels, out, **kw):
    pairs = [(run(n), l) for n, l in zip(names, labels) if (run(n) / "summary.json").exists()]
    if pairs:
        plot_curves([p for p, _ in pairs], [l for _, l in pairs], kw.pop("metric", "test_acc"), FIG / out, **kw)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plot_schedule(FIG / "schedules.png")

    # Fig. 1a: super-convergence vs the paper's long piecewise-constant baseline.
    curves([clr(10000), pc(80000)], ["CLR 0.1-3, 10k it", "PC 0.35 + warmup, 80k it"],
           "fig1a_clr10k_vs_pc80k.png", title="ResNet-56, CIFAR-10: CLR vs piecewise constant (Fig. 1a)")

    # Equal-budget control.
    curves([clr(10000), pc(10000), clr(20000), pc(20000)],
           ["CLR, 10k it", "PC, 10k it", "CLR, 20k it", "PC, 20k it"],
           "equal_budget_curves.png", title="Equal iteration budget: CLR vs piecewise constant")

    # Fig. 1b / Table 6: cycle length.
    lengths = [2000, 4000, 6000, 8000, 10000, 20000]
    curves([clr(i) for i in lengths], [f"{i // 1000}k it" for i in lengths], "fig1b_cycle_length.png",
           title="CLR 0.1-3 with different lengths (Fig. 1b)")
    clr10_mean, clr10_std, _ = mean_std([clr(10000, seed=s) for s in range(3)])
    pc10_mean, pc10_std, _ = mean_std([pc(10000, seed=s) for s in range(3)])
    ours = [final_acc(clr(i)) for i in lengths]
    ours[lengths.index(10000)] = clr10_mean
    pc_ours = [None, None, None, None, pc10_mean, final_acc(pc(20000))]
    plot_sweep([i / 1000 for i in lengths],
               {"Ours, CLR": ours, "Paper, CLR (Table 6, Fig. 1b)": [89.7, 91.1, 92.1, 91.7, 92.4, 92.7],
                "Ours, PC (same budget)": pc_ours},
               FIG / "sweep_cycle_length.png", "Training length, thousands of iterations",
               "Final accuracy vs training length",
               errors={"Ours, CLR": [None, None, None, None, clr10_std, None],
                       "Ours, PC (same budget)": [None, None, None, None, pc10_std, None]})

    # Fig. 2b / 8a: LR range test.
    ranges = existing("range_lr0-3_it20000_bs1000_wd0.0001_s0", "range_lr0-3_it5000_bs1000_wd0.0001_s0")
    if ranges:
        plot_range(ranges, [r.name.split("_")[4].replace("it", "") + " it" for r in ranges],
                   FIG / "fig2b_8a_range_test.png", title="LR range test, ResNet-56, CIFAR-10")

    # Fig. 12b: weight decay.
    wds = ["0.001", "0.0001", "1e-05", "1e-06"]
    curves([clr(10000, wd=w) for w in wds], [f"WD {float(w):g}" for w in wds], "fig12b_weight_decay.png",
           title="CLR 0.1-3, 10k it: weight decay (Fig. 12b)")

    # Table 6: peak learning rate.
    max_lrs = ["1", "1.5", "2", "2.5", "3", "3.5"]
    ours = [final_acc(clr(10000, max_lr=m)) for m in max_lrs]
    ours[max_lrs.index("3")] = clr10_mean
    plot_sweep([float(m) for m in max_lrs],
               {"Ours": ours, "Paper (Table 6)": [91.3, 90.9, 91.7, 92.3, 92.4, 92.1]},
               FIG / "sweep_max_lr.png", "Peak learning rate (max_lr)", "CLR 0.1-max_lr, 10k it",
               errors={"Ours": [None, None, None, None, clr10_std, None]})

    # Table 1: limited data.
    ns = [10000, 20000, 30000, 40000, 50000]
    ours_clr = [final_acc(clr(10000, n=n)) for n in ns[:-1]] + [clr10_mean]
    ours_pc = [final_acc(pc(80000, n=n)) for n in ns[:-1]] + [final_acc(pc(80000))]
    ours_pc10 = [final_acc(pc(10000, n=n)) for n in ns[:-1]] + [pc10_mean]
    plot_sweep([n / 1000 for n in ns],
               {"Ours, CLR 10k it": ours_clr, "Paper, CLR 10k it": [80.6, 87.9, 89.6, 91.1, 92.4],
                "Ours, PC 80k it": ours_pc, "Paper, PC 80k it": [71.4, 82.7, 85.7, 89.1, 91.2],
                "Ours, PC 10k it (same budget)": ours_pc10},
               FIG / "sweep_train_subset.png", "Training samples, thousands",
               "Limited training data (Table 1)")

    # Text summary for notes.md.
    print("\nCLR 10k, 3 seeds: %.2f ± %.2f;  PC 10k, 3 seeds: %.2f ± %.2f"
          % (clr10_mean, clr10_std, pc10_mean, pc10_std))
    for label, name in [("CLR 20k", clr(20000)), ("PC 20k", pc(20000)), ("PC 80k", pc(80000)),
                        ("He et al. baseline", HE)]:
        print(f"{label}: {final_acc(name)}")


if __name__ == "__main__":
    main()
