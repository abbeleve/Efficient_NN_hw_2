"""Iteration-based training for the super-convergence experiments.

Every run writes to results/<name>/:
    config.json   all arguments
    metrics.csv   lr, momentum, train/test loss and accuracy every --eval-every iterations
    lr_est.csv    Eq. (8) learning-rate estimates (only with --lr-est)
    summary.json  final/best test accuracy, divergence flag, wall time

Examples:
    # super-convergence run from Fig. 1a: CLR 0.1-3, one cycle over 10k iterations
    python -m hw_2.train --schedule onecycle --lr 0.1 --max-lr 3 --iters 10000
    # piecewise-constant baseline from Fig. 1a
    python -m hw_2.train --schedule piecewise --lr 0.35 --iters 80000
    # LR range test from Fig. 2b
    python -m hw_2.train --schedule range --lr 0 --max-lr 3 --iters 5000 --eval-every 100
"""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from hw_2.data import DATASETS, GpuData
from hw_2.lr_estimate import WeightSecantLR
from hw_2.models import MODELS, get_model
from hw_2.schedules import Constant, LRRange, OneCycle, Piecewise

SCHEDULES = ("onecycle", "piecewise", "range", "const")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data and model")
    g.add_argument("--dataset", default="cifar10", choices=DATASETS)
    g.add_argument("--data-root", default="datasets")
    g.add_argument("--model", default="resnet56", choices=sorted(MODELS))
    g.add_argument("--train-subset", type=int, default=None,
                   help="train on a class-balanced subset of this many samples (Table 1)")
    g.add_argument("--subset-seed", type=int, default=0)
    g.add_argument("--no-augment", action="store_true")

    g = p.add_argument_group("schedule")
    g.add_argument("--schedule", default="onecycle", choices=SCHEDULES)
    g.add_argument("--iters", type=int, default=10_000)
    g.add_argument("--batch-size", type=int, default=1000)
    g.add_argument("--lr", type=float, default=0.1,
                   help="onecycle: minimum lr; piecewise/const: base lr; range: starting lr")
    g.add_argument("--max-lr", type=float, default=3.0, help="onecycle: peak lr; range: final lr")
    g.add_argument("--tail-frac", type=float, default=0.0,
                   help="onecycle: fraction of iterations after the cycle (annealing to --final-lr)")
    g.add_argument("--final-lr", type=float, default=None, help="onecycle: lr at the end of the tail")
    g.add_argument("--milestones", type=float, nargs="+", default=[0.5, 0.75],
                   help="piecewise: lr drops (fractions of --iters if <= 1, else iterations)")
    g.add_argument("--gamma", type=float, default=0.1, help="piecewise: lr multiplier at each drop")
    g.add_argument("--warmup-iters", type=int, default=0,
                   help="piecewise/const: linear lr warmup (needed for lr=0.35 from scratch)")

    g = p.add_argument_group("optimizer")
    g.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    g.add_argument("--momentum", type=float, default=0.9,
                   help="momentum (Adam: beta1); the cycle maximum if --momentum-min is given")
    g.add_argument("--momentum-min", type=float, default=None,
                   help="onecycle: enables cyclical momentum between this and --momentum")
    g.add_argument("--nesterov", action="store_true")
    g.add_argument("--wd", type=float, default=1e-4, help="weight decay (L2, applied to all parameters)")
    g.add_argument("--bn-momentum", type=float, default=0.05,
                   help="BatchNorm running-stats momentum; 0.05 = Caffe moving_average_fraction 0.95")

    g = p.add_argument_group("logging")
    g.add_argument("--eval-every", type=int, default=200)
    g.add_argument("--lr-est", action="store_true", help="log the Eq. (8) learning-rate estimate")
    g.add_argument("--lr-est-every", type=int, default=1, help="snapshot spacing for --lr-est")
    g.add_argument("--lr-est-iters", type=int, default=None, help="stop estimating after this many iterations")
    g.add_argument("--out", default="results")
    g.add_argument("--name", default=None, help="run directory name (default: built from the arguments)")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--save-model", action="store_true", help="save final weights to model.pt")

    g = p.add_argument_group("runtime")
    g.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_schedule(args: argparse.Namespace):
    if args.schedule == "onecycle":
        return OneCycle(args.iters, args.lr, args.max_lr, args.tail_frac, args.final_lr,
                        mom_max=args.momentum if args.momentum_min is not None else None,
                        mom_min=args.momentum_min)
    if args.schedule == "piecewise":
        return Piecewise(args.iters, args.lr, tuple(args.milestones), args.gamma, args.warmup_iters)
    if args.schedule == "range":
        return LRRange(args.iters, args.lr, args.max_lr)
    return Constant(args.iters, args.lr, args.warmup_iters)


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999),
                                weight_decay=args.wd)
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                           nesterov=args.nesterov, weight_decay=args.wd)


def set_hparams(opt: torch.optim.Optimizer, lr: float, momentum: float | None) -> None:
    for group in opt.param_groups:
        group["lr"] = lr
        if momentum is not None:
            if "betas" in group:
                group["betas"] = (momentum, group["betas"][1])
            else:
                group["momentum"] = momentum


def default_run_name(args: argparse.Namespace) -> str:
    lr = f"lr{args.lr:g}"
    if args.schedule in ("onecycle", "range"):
        lr += f"-{args.max_lr:g}"
    parts = [args.dataset, args.model, args.schedule, lr, f"it{args.iters}", f"bs{args.batch_size}",
             f"wd{args.wd:g}"]
    if args.warmup_iters and args.schedule in ("piecewise", "const"):
        parts.append(f"wu{args.warmup_iters}")
    if args.train_subset:
        parts.append(f"n{args.train_subset}")
    if args.optimizer != "sgd":
        parts.append(args.optimizer)
    parts.append(f"s{args.seed}")
    return "_".join(parts)


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: GpuData, autocast) -> tuple[float, float]:
    model.eval()
    loss_sum, correct = torch.zeros((), device=data.device), torch.zeros((), device=data.device)
    for x, y in data.test_batches():
        with autocast():
            out = model(x.contiguous(memory_format=torch.channels_last))
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum")
        correct += (out.argmax(1) == y).sum()
    model.train()
    n = len(data.y_test)
    return loss_sum.item() / n, correct.item() / n


class CsvLog:
    def __init__(self, path: Path, fields: list[str]):
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def main(argv=None) -> dict:
    args = parse_args(argv)
    run_dir = Path(args.out) / (args.name or default_run_name(args))
    if (run_dir / "summary.json").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} already finished; pass --overwrite to rerun")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    data = GpuData(args.dataset, args.data_root, train_subset=args.train_subset,
                   subset_seed=args.subset_seed, augment=not args.no_augment)
    model = get_model(args.model, data.num_classes, args.bn_momentum)
    model = model.to(data.device, memory_format=torch.channels_last)
    opt = build_optimizer(args, model)
    schedule = build_schedule(args)
    gen = torch.Generator(device=data.device)
    gen.manual_seed(args.seed)
    batches = data.train_batches(args.batch_size, gen)
    autocast = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp == "bf16"
                else nullcontext)

    metrics = CsvLog(run_dir / "metrics.csv",
                     ["iter", "epoch", "lr", "momentum", "train_loss", "train_acc",
                      "test_loss", "test_acc", "time_s"])
    estimator, est_log = None, None
    if args.lr_est:
        estimator = WeightSecantLR(model.parameters(), every=args.lr_est_every)
        estimator.start()
        est_log = CsvLog(run_dir / "lr_est.csv",
                         ["iter", "lr", "lr_used", "est_l1", "est_l1_ema", "est_l2", "est_l2_ema"])

    print(f"{run_dir.name}: {data.train_size} train samples, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    loss_sum = torch.zeros((), device=data.device)
    correct_sum = torch.zeros((), device=data.device)
    seen, window_batches = 0, 0
    best_acc, best_iter, diverged = 0.0, 0, False
    test_acc = float("nan")
    start = time.perf_counter()

    for it in range(args.iters):
        lr, momentum = schedule(it)
        set_hparams(opt, lr, momentum)
        x, y = next(batches)
        with autocast():
            out = model(x.contiguous(memory_format=torch.channels_last))
            loss = F.cross_entropy(out.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_sum += loss.detach()
        correct_sum += (out.argmax(1) == y).sum()
        seen += len(y)
        window_batches += 1

        if estimator is not None and (args.lr_est_iters is None or it < args.lr_est_iters):
            est = estimator.step(lr)
            if est is not None:
                est_log.write({"iter": it + 1, "lr": lr, **est})

        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
            train_loss = loss_sum.item() / window_batches
            train_acc = correct_sum.item() / seen
            if not math.isfinite(train_loss):
                diverged = True
                print(f"[{it + 1}] diverged (train loss {train_loss}) at lr {lr:.4g}", flush=True)
                break
            test_loss, test_acc = evaluate(model, data, autocast)
            elapsed = time.perf_counter() - start
            current_mom = opt.param_groups[0].get("momentum", opt.param_groups[0].get("betas", [None])[0])
            metrics.write({"iter": it + 1, "epoch": round((it + 1) * args.batch_size / data.train_size, 3),
                           "lr": lr, "momentum": current_mom, "train_loss": train_loss,
                           "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc,
                           "time_s": round(elapsed, 1)})
            if test_acc > best_acc:
                best_acc, best_iter = test_acc, it + 1
            print(f"[{it + 1:>6}/{args.iters}] lr {lr:.4f}  train loss {train_loss:.4f} acc {train_acc:.4f}"
                  f"  test loss {test_loss:.4f} acc {test_acc:.4f}  {elapsed:.0f}s", flush=True)
            loss_sum.zero_()
            correct_sum.zero_()
            seen, window_batches = 0, 0

    metrics.close()
    if est_log is not None:
        est_log.close()
    if args.save_model:
        torch.save(model.state_dict(), run_dir / "model.pt")
    summary = {"final_test_acc": None if diverged else test_acc, "best_test_acc": best_acc,
               "best_iter": best_iter, "diverged": diverged, "iters_done": it + 1,
               "wall_time_s": round(time.perf_counter() - start, 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done: final test acc {summary['final_test_acc']}, best {best_acc:.4f} @ {best_iter}")
    return summary


if __name__ == "__main__":
    main()
