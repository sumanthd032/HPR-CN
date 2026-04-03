"""
Performance Metrics
====================
Computes aggregate statistics from a simulation run for use in paper tables.
"""

import numpy as np
from typing import List

from hpr.network  import NetworkStats
from hpr.quality  import estimate_video_quality


def compute_metrics(stats: List[NetworkStats]) -> dict:
    """
    Aggregate performance metrics over a complete simulation run.

    Metrics
    -------
    mae_kbps            : Mean Absolute Error of bandwidth estimate.
    rmse_kbps           : Root Mean Square Error of bandwidth estimate.
    mape_percent        : Mean Absolute Percentage Error (%).
    utilization         : Avg sending rate / avg available bandwidth.
    overestimation_pct  : Avg positive overshoot as % of actual bandwidth
                          (non-zero → sender causes extra congestion).
    avg_delay_ms        : Mean one-way packet delay (ms).
    avg_loss_rate       : Mean packet loss rate (0–1).
    avg_psnr_db         : Mean Peak Signal-to-Noise Ratio (dB).
    avg_mos             : Mean Opinion Score (1–5, higher = better quality).
    avg_convergence_ms  : Avg time (ms) to re-converge within 20 % of actual
                          after a bandwidth change > 200 kbps.
    """
    actuals   = np.array([s.actual_bw_kbps     for s in stats])
    estimates = np.array([s.estimated_bw_kbps  for s in stats])
    rates     = np.array([s.sending_rate_kbps  for s in stats])
    delays    = np.array([s.delay_ms           for s in stats])
    losses    = np.array([s.loss_rate          for s in stats])

    safe_actuals = np.maximum(actuals, 1.0)

    # ---- Estimation accuracy ----
    errors     = estimates - actuals
    mae        = float(np.mean(np.abs(errors)))
    rmse       = float(np.sqrt(np.mean(errors ** 2)))
    mape       = float(np.mean(np.abs(errors) / safe_actuals) * 100)

    # ---- Throughput efficiency ----
    utilization       = float(np.mean(rates / safe_actuals))
    overestimation    = float(np.mean(np.maximum(estimates - actuals, 0) / safe_actuals) * 100)

    # ---- Video quality ----
    quality_scores = [estimate_video_quality(r, a) for r, a in zip(rates, actuals)]
    avg_psnr       = float(np.mean([q["psnr"] for q in quality_scores]))
    avg_mos        = float(np.mean([q["mos"]  for q in quality_scores]))

    # ---- Convergence time ----
    # For each significant bandwidth step (> 200 kbps change), measure how many
    # steps until the estimate is within 20 % of actual, then convert to ms.
    interval_ms       = (stats[1].time_ms - stats[0].time_ms) if len(stats) > 1 else 100
    convergence_times = []
    for i in range(1, len(actuals)):
        if abs(actuals[i] - actuals[i - 1]) > 200:
            for j in range(i, min(i + 100, len(actuals))):
                if abs(estimates[j] - actuals[j]) / safe_actuals[j] < 0.20:
                    convergence_times.append((j - i) * interval_ms)
                    break

    avg_convergence = float(np.mean(convergence_times)) if convergence_times else 0.0

    return {
        "mae_kbps":           round(mae,             2),
        "rmse_kbps":          round(rmse,            2),
        "mape_percent":       round(mape,            2),
        "utilization":        round(utilization,     4),
        "overestimation_pct": round(overestimation,  2),
        "avg_delay_ms":       round(float(np.mean(delays)), 2),
        "avg_loss_rate":      round(float(np.mean(losses)), 4),
        "avg_psnr_db":        round(avg_psnr,        2),
        "avg_mos":            round(avg_mos,         2),
        "avg_convergence_ms": round(avg_convergence, 2),
    }
