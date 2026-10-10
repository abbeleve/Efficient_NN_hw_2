"""Learning-rate estimates from local curvature (paper Section 4).

WeightSecantLR implements the paper's simplified Hessian-free estimate,
Eq. (8), computed from the weights of three consecutive snapshots:

    eps* = eps * sum|theta_{i+1} - theta_i| / sum|2 theta_{i+1} - theta_i - theta_{i+2}|

The derivation substitutes g_i = (theta_i - theta_{i+1}) / eps into the secant
estimate eps* = dtheta / dg (Eq. 7), so it is exact only for plain gradient
descent; with momentum or minibatch noise it is an approximation, as in the
paper, which ran it on momentum-SGD weights.
"""

from collections import deque

import torch


class WeightSecantLR:
    """Paper Eq. (8), aggregated over all weights as in Schaul et al. (2013).

    Logs two aggregations: "l1" (sum of absolute values, the paper's choice)
    and "l2" (square root of the sum of squares, which the paper says gives
    similar results), each with an exponential moving average
    LR <- alpha * eps* + (1 - alpha) * LR, alpha = 0.1 as in the paper.

    With every=k, snapshots are taken every k iterations and k steps are
    treated as one step whose learning rate is the sum of the k per-step rates.
    Using the per-step rate instead would make the estimate k times too small
    (on a quadratic it would give 1/(k h) instead of 1/h).
    """

    def __init__(self, params, every: int = 1, alpha: float = 0.1):
        self.params = [p for p in params if p.requires_grad]
        self.every = every
        self.alpha = alpha
        self.snapshots: deque[torch.Tensor] = deque(maxlen=3)
        self.lr_sums: deque[float] = deque(maxlen=2)
        self.ema: dict[str, float] = {}
        self._lr_acc = 0.0
        self._steps = 0

    @torch.no_grad()
    def _flat_weights(self) -> torch.Tensor:
        return torch.cat([p.detach().float().flatten() for p in self.params])

    def start(self) -> None:
        """Records theta_0; call once before the first optimizer step."""
        self.snapshots.append(self._flat_weights())

    @torch.no_grad()
    def step(self, lr: float) -> dict[str, float] | None:
        """Call after each optimizer step taken with learning rate `lr`.

        Returns the estimates whenever a new snapshot completes a triple.
        """
        self._lr_acc += lr
        self._steps += 1
        if self._steps % self.every:
            return None
        self.snapshots.append(self._flat_weights())
        self.lr_sums.append(self._lr_acc)
        self._lr_acc = 0.0
        if len(self.snapshots) < 3:
            return None

        theta0, theta1, theta2 = self.snapshots
        step = theta1 - theta0
        step_change = 2 * theta1 - theta0 - theta2
        eps = sum(self.lr_sums) / 2  # lr of one (k-iteration) step, averaged over the two steps
        out = {"lr_used": eps}
        for norm, ord_ in (("l1", 1), ("l2", 2)):
            num = torch.linalg.vector_norm(step, ord_)
            den = torch.linalg.vector_norm(step_change, ord_)
            est = (eps * num / den).item() if den > 0 else float("nan")
            prev = self.ema.get(norm)
            if est == est:  # skip NaN in the moving average
                self.ema[norm] = est if prev is None else self.alpha * est + (1 - self.alpha) * prev
            out[f"est_{norm}"] = est
            out[f"est_{norm}_ema"] = self.ema.get(norm, float("nan"))
        return out
