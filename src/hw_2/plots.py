"""Figures and tables from run directories written by hw_2.train.

    python -m hw_2.plots curves  RUN [RUN ...] [--metric test_acc] [--labels ...] -o fig.png
    python -m hw_2.plots range   RUN [RUN ...] -o fig.png      # LR range test (Fig. 2b / 8a)
    python -m hw_2.plots lrest   RUN [RUN ...] -o fig.png      # Eq. (8) estimates (Fig. 3)
    python -m hw_2.plots schedule -o fig.png                    # LR / momentum shapes for slides
    python -m hw_2.plots table   RUN [RUN ...]                  # markdown table of results
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated for colour-vision deficiency on adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "legend.labelcolor": INK,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})

METRIC_LABELS = {"test_acc": "Test accuracy", "train_acc": "Train accuracy", "test_loss": "Test loss",
                 "train_loss": "Train loss", "lr": "Learning rate", "momentum": "Momentum"}


def read_csv(path: Path) -> dict[str, list[float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) if r[k] not in ("", "None") else float("nan") for r in rows] for k in rows[0]}


def _labels(runs: list[Path], labels: list[str] | None) -> list[str]:
    if labels and len(labels) != len(runs):
        raise SystemExit(f"got {len(labels)} labels for {len(runs)} runs")
    return labels or [r.name for r in runs]


def _save(fig, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


def plot_curves(runs, labels, metric, out, title=None, x="iter"):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (run, label) in enumerate(zip(runs, labels)):
        m = read_csv(run / "metrics.csv")
        color = SERIES[i % len(SERIES)]
        ax.plot(m[x], m[metric], color=color, label=label)
        if len(runs) <= 2:  # end labels collide with more series; the legend and tables carry them
            ax.annotate(f"{m[metric][-1]:.3f}", (m[x][-1], m[metric][-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK_MUTED)
    ax.set_xlabel("Iteration" if x == "iter" else x.capitalize())
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    if metric.endswith("acc"):
        ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
    ax.set_title(title or METRIC_LABELS.get(metric, metric), loc="left")
    if len(runs) > 1:
        ax.legend(loc="best")
    _save(fig, out)


def plot_range(runs, labels, out, title=None):
    """Two panels sharing the LR axis: test accuracy (Fig. 2b) and train/test loss (Fig. 8a)."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (run, label) in enumerate(zip(runs, labels)):
        m = read_csv(run / "metrics.csv")
        ax_acc.plot(m["lr"], m["test_acc"], color=SERIES[i % len(SERIES)], label=label)
    ax_acc.set(xlabel="Learning rate", ylabel="Test accuracy")
    ax_acc.set_title("Test accuracy", loc="left")
    if len(runs) > 1:
        ax_acc.legend(loc="lower right")

    # Loss panel uses the first run only, so train and test take slots 1 and 2.
    m = read_csv(runs[0] / "metrics.csv")
    ax_loss.plot(m["lr"], m["train_loss"], color=SERIES[0], label="Train loss")
    ax_loss.plot(m["lr"], m["test_loss"], color=SERIES[1], label="Test loss")
    ax_loss.set(xlabel="Learning rate", ylabel="Cross-entropy")
    ax_loss.set_title(f"Loss ({labels[0]})" if len(runs) > 1 else "Loss", loc="left")
    ax_loss.legend(loc="upper right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    _save(fig, out)


def plot_lrest(runs, labels, out, norm="l1", ema=True, title=None, max_iter=None):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    col = f"est_{norm}_ema" if ema else f"est_{norm}"
    for i, (run, label) in enumerate(zip(runs, labels)):
        e = read_csv(run / "lr_est.csv")
        pts = [(it, v) for it, v in zip(e["iter"], e[col]) if max_iter is None or it <= max_iter]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[i % len(SERIES)], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Estimated learning rate")
    ax.set_title(title or f"Eq. (8) estimate, {norm.upper()}{', EMA 0.1' if ema else ''}", loc="left")
    if len(runs) > 1:
        ax.legend(loc="upper right")
    _save(fig, out)


def plot_sweep(x, series: dict[str, list], out, xlabel, title, errors: dict[str, list] | None = None,
               xlog=False, ylabel="Final test accuracy, %"):
    """Final metric against a swept parameter, e.g. ours vs. paper; None values are skipped."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (label, ys) in enumerate(series.items()):
        pts = [(xi, yi, (errors or {}).get(label, [None] * len(x))[j])
               for j, (xi, yi) in enumerate(zip(x, ys)) if yi is not None]
        xs, vals, errs = zip(*pts)
        color = SERIES[i % len(SERIES)]
        ax.plot(xs, vals, color=color, marker="o", markersize=5, label=label)
        if any(e is not None for e in errs):
            ax.errorbar(xs, vals, yerr=[e or 0 for e in errs], fmt="none", ecolor=color, capsize=3, lw=1)
    if xlog:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    _save(fig, out)


def plot_schedule(out):
    from hw_2.schedules import OneCycle, Piecewise

    total = 10_000
    shapes = {
        "Piecewise constant (0.35, x0.1 at 50% / 75%)": Piecewise(total, 0.35),
        "CLR 0.1-3, one cycle": OneCycle(total, 0.1, 3.0),
        "1cycle 0.1-3, 10% tail, momentum 0.95-0.85": OneCycle(total, 0.1, 3.0, tail_frac=0.1,
                                                               mom_max=0.95, mom_min=0.85),
    }
    its = range(total)
    fig, (ax_lr, ax_mom) = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (label, sched) in enumerate(shapes.items()):
        values = [sched(it) for it in its]
        ax_lr.plot(its, [v[0] for v in values], color=SERIES[i], label=label)
        ax_mom.plot(its, [v[1] if v[1] is not None else 0.9 for v in values], color=SERIES[i], label=label)
    ax_lr.set(xlabel="Iteration", ylabel="Learning rate")
    ax_lr.set_title("Learning rate", loc="left")
    ax_mom.set(xlabel="Iteration", ylabel="Momentum")
    ax_mom.set_title("Momentum (piecewise and CLR overlap at 0.9)", loc="left")
    fig.legend(*ax_lr.get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


def print_table(runs, labels):
    keys = ["dataset", "model", "schedule", "lr", "max_lr", "iters", "batch_size", "wd", "train_subset", "seed"]
    print("| run | " + " | ".join(keys) + " | final acc | best acc | diverged | time, s |")
    print("|" + "---|" * (len(keys) + 5))
    def pct(v):
        return "—" if v is None else f"{v * 100:.2f}"

    for run, label in zip(runs, labels):
        cfg = json.loads((run / "config.json").read_text())
        summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
        cells = [label] + [str(cfg.get(k, "")) for k in keys] + [
            pct(summary.get("final_test_acc")), pct(summary.get("best_test_acc")),
            str(summary.get("diverged", "")), str(summary.get("wall_time_s", ""))]
        print("| " + " | ".join(cells) + " |")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("curves", "range", "lrest", "table"):
        s = sub.add_parser(name)
        s.add_argument("runs", nargs="+", type=Path)
        s.add_argument("--labels", nargs="+")
        if name != "table":
            s.add_argument("-o", "--out", required=True)
            s.add_argument("--title")
    sub.choices["curves"].add_argument("--metric", default="test_acc")
    sub.choices["curves"].add_argument("--x", default="iter", choices=["iter", "epoch"])
    sub.choices["lrest"].add_argument("--norm", default="l1", choices=["l1", "l2"])
    sub.choices["lrest"].add_argument("--raw", action="store_true", help="plot raw estimates instead of the EMA")
    sub.choices["lrest"].add_argument("--max-iter", type=int)
    sub.add_parser("schedule").add_argument("-o", "--out", required=True)
    args = p.parse_args(argv)

    if args.cmd == "schedule":
        plot_schedule(args.out)
        return
    labels = _labels(args.runs, args.labels)
    if args.cmd == "curves":
        plot_curves(args.runs, labels, args.metric, args.out, args.title, args.x)
    elif args.cmd == "range":
        plot_range(args.runs, labels, args.out, args.title)
    elif args.cmd == "lrest":
        plot_lrest(args.runs, labels, args.out, args.norm, not args.raw, args.title, args.max_iter)
    else:
        print_table(args.runs, labels)


if __name__ == "__main__":
    main()
