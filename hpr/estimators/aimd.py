"""
AIMD Bandwidth Estimator (Baseline)
=====================================
Additive Increase / Multiplicative Decrease — the classic TCP congestion
control heuristic adapted for real-time media rate control.

Role in paper: lower-bound baseline showing the limitation of loss-only
feedback without any delay-gradient or predictive components.
"""

import numpy as np
from typing import List

from hpr.estimators.base import BaseEstimator
from hpr.network import Packet


class AIMDEstimator(BaseEstimator):
    """
    AIMD rate controller.

    Increase rule  : add ``additive_increase`` kbps every step if loss is low.
    Decrease rule  : halve the estimate when loss exceeds ``loss_threshold``.
    """

    def __init__(self, initial_estimate_kbps: float = 1000.0) -> None:
        super().__init__(initial_estimate_kbps)
        self.additive_increase  = 50.0    # kbps per time step
        self.multiplicative_decrease = 0.5
        self.loss_threshold     = 0.02    # 2 %

    def update(self, packets: List[Packet], current_time_ms: float) -> float:
        if not packets:
            return self.estimate

        lost       = sum(1 for p in packets if p.lost)
        loss_rate  = lost / len(packets)

        if loss_rate > self.loss_threshold:
            self.estimate *= self.multiplicative_decrease
        else:
            self.estimate += self.additive_increase

        self.estimate = float(np.clip(self.estimate, 100, 10_000))
        self.history.append(self.estimate)
        return self.estimate
