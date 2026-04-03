"""
NADA: Network-Assisted Dynamic Adaptation
==========================================
A simplified simulation model of the NADA congestion controller
standardised in RFC 8698 (Sarker et al., 2020).

Role in paper: second reactive-only baseline that unifies delay and loss
into a single synthetic congestion signal — addressing the weakness of
GCC's independent delay and loss controllers.

Key NADA mechanisms modelled here
-----------------------------------
1. **Baseline OWD tracking**: running minimum of OWD estimates the
   propagation-only delay; queuing delay deviation x_d = OWD - OWD_min.

2. **Loss congestion proxy**: high packet-loss rate is converted to an
   equivalent congestion unit x_L, preventing NADA from misinterpreting
   wireless random losses as queue congestion (in practice, RFC 8698
   leaves this mapping to the implementation).

3. **Synthetic congestion signal**: x_n = x_d + x_L unifies both signals.

4. **Accelerated ramp-up**: when x_n is well below the reference threshold
   and loss is absent, NADA probes upward aggressively.

5. **Gradual AIMD update**: when x_n exceeds the reference, the rate
   decreases proportionally to the congestion overshoot; otherwise it
   increases mildly.

NOTE: This is an academic simulation model.  It captures the principal
mechanisms of RFC 8698 without reproducing any proprietary implementation.
Several details (FEC, receiver-driven feedback format, multi-flow fairness
proofs) are outside scope for a single-flow packet simulator.
"""

import numpy as np
from typing import List

from hpr.estimators.base import BaseEstimator
from hpr.network import Packet


class NADAEstimator(BaseEstimator):
    """
    NADA bandwidth estimator (RFC 8698, simplified simulation model).

    Parameters
    ----------
    initial_estimate_kbps : float  Starting estimate (kbps).
    x_ref_ms              : float  Reference queuing-delay threshold (ms).
                                   RFC 8698 default ≈ 50 ms.
    """

    def __init__(
        self,
        initial_estimate_kbps: float = 1000.0,
        x_ref_ms: float = 50.0,
    ) -> None:
        super().__init__(initial_estimate_kbps)

        # ---- OWD baseline ----
        self.owd_min: float = float("inf")   # running OWD minimum (propagation baseline)
        self.avg_x_d: float = 0.0            # EWMA of queuing delay deviation
        self.alpha:   float = 0.85           # EWMA smoothing coefficient

        # ---- NADA thresholds ----
        self.x_ref: float = x_ref_ms         # reference congestion threshold (ms)
        self.loss_proxy_scale: float = 500.0  # maps loss rate → ms-equivalent units

        # ---- Ramp-up state ----
        self.ramp_up_steps: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_owd_and_loss(
        self, packets: List[Packet]
    ):
        """Return (mean_owd_ms, loss_rate) from the current packet batch."""
        received  = [p for p in packets if not p.lost]
        loss_rate = (len(packets) - len(received)) / max(len(packets), 1)

        if len(received) < 2:
            return None, loss_rate

        owds     = [p.recv_time_ms - p.send_time_ms for p in received]
        mean_owd = float(np.mean(owds))
        return mean_owd, loss_rate

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, packets: List[Packet], current_time_ms: float) -> float:
        if not packets:
            return self.estimate

        mean_owd, loss_rate = self._compute_owd_and_loss(packets)

        if mean_owd is None:
            # Too few received packets — slow cautious ramp
            self.estimate = float(np.clip(self.estimate * 1.02, 100, 10_000))
            self.history.append(self.estimate)
            return self.estimate

        # ---- Step 1: update OWD baseline (propagation-only floor) ----
        if mean_owd < self.owd_min:
            self.owd_min = mean_owd
        # Allow baseline to drift upward very slowly (path route changes)
        self.owd_min += 0.0005 * (mean_owd - self.owd_min)

        # ---- Step 2: queuing delay deviation ----
        x_d = max(0.0, mean_owd - self.owd_min)
        self.avg_x_d = self.alpha * self.avg_x_d + (1.0 - self.alpha) * x_d

        # ---- Step 3: loss congestion proxy ----
        # RFC 8698 does not fix a specific loss → delay mapping; we use
        # a linear scaling calibrated so 4 % loss ≈ x_ref (50 ms).
        x_L = loss_rate * self.loss_proxy_scale

        # ---- Step 4: synthetic congestion signal ----
        x_n = self.avg_x_d + x_L

        # ---- Step 5: rate update ----
        if x_n < self.x_ref * 0.5 and loss_rate < 0.02:
            # Accelerated ramp-up: probe aggressively when channel is clear
            self.ramp_up_steps += 1
            delta = max(50.0, self.estimate * 0.08)
            self.estimate = min(self.estimate + delta, 10_000.0)
        else:
            self.ramp_up_steps = 0
            if x_n > self.x_ref:
                # Gradual decrease proportional to congestion overshoot
                # Mirrors RFC 8698 multiplicative decrease with AIMD coefficient
                overshoot = x_n / self.x_ref          # > 1 when congested
                decrease  = min(0.50, 0.12 * overshoot)
                self.estimate *= (1.0 - decrease)
            else:
                # Below threshold but not in full ramp: mild additive increase
                delta = max(10.0, self.estimate * 0.02)
                self.estimate = min(self.estimate + delta, 10_000.0)

        self.estimate = float(np.clip(self.estimate, 100, 10_000))
        self.history.append(self.estimate)
        return self.estimate
