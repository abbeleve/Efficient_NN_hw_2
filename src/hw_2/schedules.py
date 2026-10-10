"""Learning-rate (and momentum) schedules as pure functions of the iteration.

Each schedule maps iteration `it` in [0, total) to (lr, momentum); momentum is
None when the schedule leaves it untouched. They are applied by hand every
iteration, which keeps the exact shapes from the paper easy to plot and check.
"""

from dataclasses import dataclass


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass(frozen=True)
class OneCycle:
    """CLR / 1cycle from the paper (Section 3).

    lr_min -> lr_max over `step_size` iterations, back to lr_min over another
    `step_size`, then lr_min -> lr_final for the remaining iterations. With
    tail_frac=0 this is a single CLR cycle (step_size = total/2), which is what
    the ResNet-56/Cifar-10 experiments use ("CLR 0.1-3, stepsize 5000" for 10k
    iterations). Table 2 runs keep ~10% of training for the tail.

    If mom_max/mom_min are set, momentum cycles in antiphase to the learning
    rate (mom_max -> mom_min -> mom_max) and stays at mom_max in the tail.
    """

    total: int
    lr_min: float
    lr_max: float
    tail_frac: float = 0.0
    lr_final: float | None = None
    mom_max: float | None = None
    mom_min: float | None = None

    @property
    def step_size(self) -> int:
        return int(round(self.total * (1 - self.tail_frac) / 2))

    def __call__(self, it: int) -> tuple[float, float | None]:
        step = self.step_size
        lr_final = self.lr_final if self.lr_final is not None else self.lr_min * 1e-3
        cyclic_mom = self.mom_max is not None and self.mom_min is not None
        if it < step:
            t = it / step
            lr = _lerp(self.lr_min, self.lr_max, t)
            mom = _lerp(self.mom_max, self.mom_min, t) if cyclic_mom else None
        elif it < 2 * step:
            t = (it - step) / step
            lr = _lerp(self.lr_max, self.lr_min, t)
            mom = _lerp(self.mom_min, self.mom_max, t) if cyclic_mom else None
        else:
            tail = max(self.total - 2 * step, 1)
            lr = _lerp(self.lr_min, lr_final, (it - 2 * step) / tail)
            mom = self.mom_max if cyclic_mom else None
        return lr, mom


def _warmup(lr: float, it: int, warmup_iters: int) -> float:
    """Linear warmup from lr/warmup_iters to lr (Goyal et al., 2017)."""
    return lr * (it + 1) / warmup_iters if it < warmup_iters else lr


@dataclass(frozen=True)
class Piecewise:
    """The paper's "typical" regime: constant lr, multiplied by gamma at each milestone.

    Milestones <= 1 are fractions of `total`, larger values are iterations.
    Starting ResNet-56 from scratch at lr=0.35 with batch 1000 kills the network
    (loss stuck at ln(10)), so a short linear warmup can be enabled.
    """

    total: int
    lr: float
    milestones: tuple[float, ...] = (0.5, 0.75)
    gamma: float = 0.1
    warmup_iters: int = 0

    def __call__(self, it: int) -> tuple[float, float | None]:
        bounds = [m * self.total if m <= 1 else m for m in self.milestones]
        drops = sum(it >= b for b in bounds)
        return _warmup(self.lr * self.gamma ** drops, it, self.warmup_iters), None


@dataclass(frozen=True)
class LRRange:
    """LR range test: linear increase from lr_start to lr_end over the whole run."""

    total: int
    lr_start: float
    lr_end: float

    def __call__(self, it: int) -> tuple[float, float | None]:
        return _lerp(self.lr_start, self.lr_end, it / max(self.total - 1, 1)), None


@dataclass(frozen=True)
class Constant:
    total: int
    lr: float
    warmup_iters: int = 0

    def __call__(self, it: int) -> tuple[float, float | None]:
        return _warmup(self.lr, it, self.warmup_iters), None
