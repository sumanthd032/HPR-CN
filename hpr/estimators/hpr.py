"""
HPR: Hybrid Predictive-Reactive Bandwidth Estimator
======================================================
Novel algorithm proposed for real-time video conferencing bandwidth estimation.

Architecture (three-layer fusion)
----------------------------------
Layer 1 — Reactive  : delay-gradient EWMA + congestion-type classification
Layer 2 — Predictive: exponentially-weighted linear trend extrapolation
Layer 3 — Fusion    : Kalman filter with *adaptive* process noise Q

Key novelty over GCC-based approaches
--------------------------------------
1. **Congestion-type classification**: distinguishes between
   - ``delay``   — queue build-up only (gentle back-off preserves throughput)
   - ``severe``  — high delay + high loss (aggressive back-off)
   - ``wireless``— loss without delay growth (random errors, not queue)
   - ``mild``    — borderline hints (cautious increase)
   - ``clear``   — no congestion signal (aggressive probing)

2. **Adaptive Kalman process noise (Q)**: Q scales with the coefficient of
   variation (CV) of recent measurements.  A volatile link gets a larger Q,
   so the filter trusts new measurements more.  A stable link gets a smaller
   Q, smoothing out noise.  Static Q values (used in prior work) are a
   special case with CV = 0.

3. **Exponentially-weighted least-squares trend prediction**: recent samples
   carry more weight, reducing the influence of stale history on the slope
   estimate — more responsive to trend reversals than ordinary OLS.

4. **Preemptive rate reduction**: when the trend predicts a future drop and
   the channel is currently stable (CV is low), HPR reduces rate *before*
   congestion is detected — eliminating the GCC delay-to-react.
"""

import numpy as np
from typing import List, Optional, Tuple

from hpr.estimators.base import BaseEstimator
from hpr.network import Packet


class HybridPredictiveEstimator(BaseEstimator):
    """
    Hybrid Predictive-Reactive (HPR) bandwidth estimator.

    Parameters
    ----------
    initial_estimate_kbps : float  Starting estimate (default 1 000 kbps).
    process_noise         : float  Base Kalman Q — scales with volatility.
    measurement_noise     : float  Kalman R — trust in raw measurements.
    prediction_weight     : float  Max weight given to the predictive signal
                                   (0 = fully reactive, 1 = fully predictive).
    window_size           : int    Look-back window for trend regression.
    delay_threshold       : float  Delay-gradient overuse threshold (ms).
    """

    def __init__(
        self,
        initial_estimate_kbps: float = 1000.0,
        process_noise:         float = 500.0,
        measurement_noise:     float = 120.0,
        prediction_weight:     float = 0.35,
        window_size:           int   = 10,
        delay_threshold:       float = 10.0,
        ablation_mode:         Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        ablation_mode : str or None
            Controls which HPR components are active (for ablation studies).
            ``None``            — full HPR (default)
            ``"no_predictor"``  — Layer 2 disabled; purely reactive + adaptive-Q Kalman
            ``"fixed_q"``       — Layer 3 uses fixed Q = Q0 (no adaptive scaling)
            ``"no_preemptive"`` — preemptive reduction step disabled
            ``"no_classifier"`` — Layer 1 uses a simple 2-state (overuse/clear)
                                   classifier instead of the 5-class scheme
        """
        super().__init__(initial_estimate_kbps)

        # ---- Ablation control ----
        self.ablation_mode        = ablation_mode

        # ---- Layer 1: Reactive (delay-gradient) ----
        self.alpha                = 0.85
        self.avg_delay_gradient   = 0.0
        self.delay_threshold      = delay_threshold

        # ---- Layer 2: Predictive (trend) ----
        self.window_size          = window_size
        self.prediction_weight    = prediction_weight
        self.recent_measurements: List[float] = [initial_estimate_kbps]

        # ---- Layer 3: Kalman filter ----
        self.kalman_estimate      = initial_estimate_kbps
        self.kalman_error         = 1500.0
        self.base_process_noise   = process_noise
        self.measurement_noise    = measurement_noise

        # ---- Derived state ----
        self.stability_score      = 1.0   # 1 = very stable, 0 = volatile
        self.receive_rate         = initial_estimate_kbps
        self.step_count           = 0

    # ==================================================================
    # Layer 1: Reactive signal
    # ==================================================================

    def _extract_packet_signals(
        self, packets: List[Packet]
    ) -> Tuple[float, float, float]:
        """
        Compute delay gradient, loss rate, and receive rate from a packet batch.

        Returns
        -------
        (delay_gradient_ms, loss_rate_fraction, receive_rate_kbps)
        """
        received  = [p for p in packets if not p.lost]
        loss_rate = (len(packets) - len(received)) / max(len(packets), 1)

        if len(received) >= 2:
            bits     = sum(p.size_bytes * 8 for p in received)
            span_ms  = max(1.0, received[-1].recv_time_ms - received[0].recv_time_ms)
            recv_rate = bits / span_ms
            owds      = [p.recv_time_ms - p.send_time_ms for p in received]
            delay_grad = float(np.mean(np.diff(owds)))
        elif len(received) == 1:
            recv_rate  = (received[0].size_bytes * 8) / 100.0  # rough 100-ms interval
            delay_grad = 0.0
        else:
            recv_rate  = 0.0
            delay_grad = 0.0

        return delay_grad, loss_rate, recv_rate

    def _classify_congestion(
        self, loss_rate: float
    ) -> Tuple[str, float]:
        """
        Classify the current network state and return a reactive rate target.

        Congestion types (5-class scheme, Layer 1 of HPR)
        --------------------------------------------------
        ``severe``  : delay gradient AND loss both high — queue + overflow
        ``delay``   : delay gradient high, loss low — queue building up
        ``wireless``: loss high but delay is *not* growing — random errors
        ``mild``    : borderline delay hints, low loss
        ``clear``   : no congestion signal — safe to probe upward

        Ablation: ``no_classifier`` uses a simpler GCC-like 2-state rule
        (overuse/clear) that treats delay and loss identically.

        Returns
        -------
        (congestion_type: str, reactive_estimate_kbps: float)
        """
        high_delay   = self.avg_delay_gradient > self.delay_threshold
        moderate_del = 5.0 < self.avg_delay_gradient <= self.delay_threshold
        high_loss    = loss_rate > 0.04
        mild_loss    = 0.015 < loss_rate <= 0.04

        # ---- Ablation: simple 2-state classifier (like GCC Layer 1) ----
        if self.ablation_mode == "no_classifier":
            if high_delay or high_loss:
                return "overuse", self.estimate * 0.85
            return "clear", max(self.estimate * 1.05, self.estimate + 50)

        # ---- Full 5-class classifier ----
        if high_delay and high_loss:
            return "severe",   self.estimate * 0.78
        if high_delay:
            return "delay",    self.estimate * 0.85
        if high_loss:
            # Loss without delay growth → wireless random errors (not queue)
            return "wireless", self.estimate * 0.92
        if moderate_del or mild_loss:
            return "mild",     max(self.estimate * 1.04, self.estimate + 80)
        return "clear", max(self.estimate * 1.15, self.estimate + 250)

    # ==================================================================
    # Layer 2: Predictive signal (exponentially-weighted OLS)
    # ==================================================================

    def _predict_trend(self) -> Optional[float]:
        """
        Predict bandwidth one step ahead using exponentially-weighted
        least-squares linear regression on the recent measurement window.

        Exponential weighting ensures recent measurements influence the
        slope estimate more than older ones, making the predictor responsive
        to trend reversals without discarding historical context.

        Also updates ``stability_score`` from the coefficient of variation
        of the last 5 samples.

        Returns ``None`` when there are too few samples.
        """
        if len(self.recent_measurements) < 4:
            return None

        window = np.array(self.recent_measurements[-self.window_size:], dtype=float)
        n      = len(window)
        x      = np.arange(n, dtype=float)

        # Exponential weights: most recent point has weight 1.0,
        # oldest has weight e^{-1} ≈ 0.37
        weights = np.exp(np.linspace(-1.0, 0.0, n))

        # Weighted means
        x_bar  = np.average(x,      weights=weights)
        y_bar  = np.average(window, weights=weights)

        # Weighted slope (numerator / denominator of WLS normal equation)
        numerator   = np.sum(weights * (x - x_bar) * (window - y_bar))
        denominator = np.sum(weights * (x - x_bar) ** 2)
        slope       = numerator / denominator if denominator > 1e-9 else 0.0

        # One step ahead of the last observed point
        predicted = slope * n + (y_bar - slope * x_bar)

        # ---- Stability score from coefficient of variation ----
        if len(self.recent_measurements) >= 5:
            recent  = np.array(self.recent_measurements[-5:])
            mean_bw = np.mean(recent)
            cv      = np.std(recent) / max(mean_bw, 1.0)   # dimensionless
            # stability_score → 1 when CV → 0 (very stable)
            #                 → 0 when CV → ∞ (very volatile)
            self.stability_score = 1.0 / (1.0 + cv * 5.0)

        return max(100.0, predicted)

    # ==================================================================
    # Layer 3: Kalman filter with adaptive process noise
    # ==================================================================

    def _kalman_update(self, measurement: float) -> float:
        """
        Kalman filter update step.

        Adaptive Q: process noise scales with coefficient of variation (CV)
        of recent measurements.  A volatile link gets larger Q, so the filter
        tracks faster changes; a stable link gets smaller Q, smoothing noise.
        Static-Q Kalman (prior work, and the ``fixed_q`` ablation) is a
        special case where Q remains constant at Q0.
        """
        # ---- Compute adaptive Q (disabled in fixed_q ablation) ----
        if self.ablation_mode == "fixed_q" or len(self.recent_measurements) < 5:
            adaptive_q = self.base_process_noise
        else:
            recent     = np.array(self.recent_measurements[-5:])
            cv         = np.std(recent) / max(np.mean(recent), 1.0)
            adaptive_q = self.base_process_noise * (1.0 + min(cv * 3.0, 3.0))

        # ---- Kalman predict / update ----
        predicted_error      = self.kalman_error + adaptive_q
        kalman_gain          = predicted_error / (predicted_error + self.measurement_noise)
        self.kalman_estimate = (self.kalman_estimate
                                + kalman_gain * (measurement - self.kalman_estimate))
        self.kalman_error    = (1.0 - kalman_gain) * predicted_error

        return self.kalman_estimate

    # ==================================================================
    # Main update loop
    # ==================================================================

    def update(self, packets: List[Packet], current_time_ms: float) -> float:
        if not packets:
            return self.estimate

        self.step_count += 1

        # ---- Step 1: extract raw signals ----
        delay_grad, loss_rate, recv_rate = self._extract_packet_signals(packets)
        self.avg_delay_gradient = (self.alpha * self.avg_delay_gradient
                                   + (1 - self.alpha) * delay_grad)
        if recv_rate > 0:
            self.receive_rate = recv_rate

        # ---- Step 2: reactive estimate ----
        congestion_type, reactive_est = self._classify_congestion(loss_rate)

        # ---- Step 3: predictive estimate (disabled in no_predictor ablation) ----
        trend_est = None if self.ablation_mode == "no_predictor" else self._predict_trend()

        # ---- Step 4: fuse reactive + predictive ----
        if trend_est is not None:
            w        = self.prediction_weight * self.stability_score
            combined = (1.0 - w) * reactive_est + w * trend_est
        else:
            combined = reactive_est

        # ---- Step 5: Kalman smoothing (adaptive Q) ----
        fused = self._kalman_update(combined)

        # ---- Step 6: emergency loss brake ----
        if loss_rate > 0.10:
            fused *= 0.60
        elif loss_rate > 0.05:
            fused *= 0.75

        # ---- Step 7: preemptive reduction (disabled in no_preemptive ablation) ----
        # When the trend clearly predicts a future decline, reduce now
        # while the channel is still stable — avoids the reactive delay.
        if (
            self.ablation_mode != "no_preemptive"
            and trend_est is not None
            and congestion_type in ("clear", "mild")
            and len(self.recent_measurements) > 15
            and self.stability_score > 0.5
            and trend_est < self.estimate * 0.70
        ):
            fused = min(fused, self.estimate * 0.95)   # gentle preemptive step

        self.estimate = float(np.clip(fused, 100, 10_000))
        self.recent_measurements.append(self.estimate)
        self.history.append(self.estimate)
        return self.estimate
