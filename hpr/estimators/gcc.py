"""
GCC-Like Bandwidth Estimator
================================
A simplified implementation of the delay-gradient congestion detection
approach described in Google's Congestion Control algorithm (RFC 8698 / RMCAT).

Role in paper: reactive-only comparison showing the cost of having no
predictive component — slower adaptation and higher overestimation on
sudden-drop traces.

NOTE: This is an academic simulation model. It captures the key mechanisms
of GCC (delay-gradient EWMA, overuse state machine, loss-based adjustment)
without re-producing any proprietary or copyrighted implementation.
"""

import numpy as np
from typing import List

from hpr.estimators.base import BaseEstimator
from hpr.network import Packet


class GCCEstimator(BaseEstimator):
    """
    Delay-gradient congestion controller (GCC-inspired).

    State machine
    -------------
    ``increase``  : no congestion signal → ramp up bitrate
    ``hold``      : uncertain → nudge toward measured receive rate
    ``decrease``  : overuse detected → back off

    Congestion signal: EWMA of the inter-arrival delay gradient exceeds
    ``delay_gradient_threshold`` for more than ``overuse_count_threshold``
    consecutive steps.
    """

    def __init__(self, initial_estimate_kbps: float = 1000.0) -> None:
        super().__init__(initial_estimate_kbps)

        # Delay-gradient filter
        self.alpha: float                  = 0.85   # EWMA smoothing factor
        self.avg_delay_gradient: float     = 0.0
        self.delay_gradient_threshold: float = 10.0  # ms — overuse threshold

        # State machine
        self.state: str        = "hold"
        self.overuse_counter: int = 0
        self.hold_count: int   = 0

        # Rate-change parameters
        self.increase_rate: float  = 0.05    # 5 % multiplicative increase
        self.decrease_rate: float  = 0.85    # 15 % multiplicative decrease

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _measure_receive_rate(self, received: List[Packet]) -> float:
        """Estimate receive rate (kbps) from a batch of received packets."""
        if len(received) < 2:
            return 0.0
        total_bits = sum(p.size_bytes * 8 for p in received)
        span_ms    = max(1.0, received[-1].recv_time_ms - received[0].recv_time_ms)
        return total_bits / span_ms   # kbps

    def _compute_delay_gradient(self, received: List[Packet]) -> float:
        """Return the mean inter-packet delay gradient for this batch."""
        if len(received) < 2:
            return 0.0
        owds      = [p.recv_time_ms - p.send_time_ms for p in received]
        gradients = np.diff(owds)
        return float(np.mean(gradients))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, packets: List[Packet], current_time_ms: float) -> float:
        if not packets:
            return self.estimate

        received     = [p for p in packets if not p.lost]
        lost         = len(packets) - len(received)
        loss_rate    = lost / len(packets)

        if len(received) < 2:
            # Too few packets: slow ramp-up to avoid stalling
            self.estimate = float(np.clip(self.estimate * 1.02, 100, 10_000))
            self.history.append(self.estimate)
            return self.estimate

        recv_rate    = self._measure_receive_rate(received)
        grad         = self._compute_delay_gradient(received)

        # EWMA of delay gradient
        self.avg_delay_gradient = (self.alpha * self.avg_delay_gradient
                                   + (1 - self.alpha) * grad)

        # ---- State machine ----
        if self.avg_delay_gradient > self.delay_gradient_threshold:
            self.overuse_counter += 1
            if self.overuse_counter > 2:
                self.state      = "decrease"
                self.hold_count = 0
        elif self.avg_delay_gradient < -2.0:
            self.state           = "increase"
            self.overuse_counter = 0
            self.hold_count      = 0
        else:
            self.overuse_counter = max(0, self.overuse_counter - 1)
            self.hold_count     += 1
            self.state = "increase" if self.hold_count > 5 else "hold"

        # ---- Apply rate change ----
        if self.state == "increase":
            self.estimate = max(
                self.estimate * (1 + self.increase_rate),
                self.estimate + 50,
            )
        elif self.state == "decrease":
            self.estimate *= self.decrease_rate
        else:   # hold
            # Blend toward measured receive rate
            self.estimate = 0.95 * self.estimate + 0.05 * recv_rate

        # ---- Loss-based safety adjustment ----
        if loss_rate > 0.05:
            self.estimate *= 0.80

        # Never fall below half of what we actually measured receiving
        if recv_rate > 0:
            self.estimate = max(self.estimate, recv_rate * 0.5)

        self.estimate = float(np.clip(self.estimate, 100, 10_000))
        self.history.append(self.estimate)
        return self.estimate
