"""
Time-Series Visualization
==========================
Generates time-series plots illustrating the closed-loop control:
  - True available bandwidth B_t
  - Estimates B̂_t from all four algorithms
  - Sending rates r_t = 0.9 × B̂_{t-1} (estimator-driven, NOT ground truth)
  - Congestion class labels for HPR

Usage:
    python plot_timeseries.py

Outputs: figures/ directory with PNG files for inclusion in the paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hpr import (
    TraceType, SimulationConfig, run_comparison, ALGORITHM_KEYS
)

os.makedirs("figures", exist_ok=True)

COLORS = {
    "AIMD":           "#888888",
    "GCC":            "#e07b39",
    "NADA":           "#4a90d9",
    "HPR (Proposed)": "#2ca02c",
    "True BW":        "#d62728",
}

SEED = 42


def plot_trace(trace_type: TraceType, filename: str, zoom=None):
    """Generate and save a time-series plot for one trace scenario."""
    config  = SimulationConfig(trace_type=trace_type)
    results = run_comparison(config, seed=SEED, verbose=False)

    # Time axis in seconds
    stats0   = list(results.values())[0]["stats"]
    time_s   = np.array([s.time_ms / 1000.0 for s in stats0])
    actual_bw = np.array([s.actual_bw_kbps for s in stats0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_bw, ax_sr = axes

    # ---- Top panel: true BW + estimates ----
    ax_bw.plot(time_s, actual_bw / 1000, color=COLORS["True BW"],
               lw=2, ls="--", label="True B/W", zorder=5)
    for algo in ALGORITHM_KEYS:
        stats = results[algo]["stats"]
        est   = np.array([s.estimated_bw_kbps for s in stats]) / 1000
        ax_bw.plot(time_s, est, color=COLORS[algo], lw=1.2, alpha=0.85, label=algo)

    ax_bw.set_ylabel("Bandwidth (Mbps)")
    ax_bw.set_title(f"Trace: {trace_type.value.replace('_', ' ').title()} — Estimator Output vs True Bandwidth")
    ax_bw.legend(loc="upper right", fontsize=8, ncol=2)
    ax_bw.set_ylim(bottom=0)
    ax_bw.grid(True, alpha=0.3)

    # ---- Bottom panel: sending rates ----
    for algo in ALGORITHM_KEYS:
        stats = results[algo]["stats"]
        sr    = np.array([s.sending_rate_kbps for s in stats]) / 1000
        ax_sr.plot(time_s, sr, color=COLORS[algo], lw=1.0, alpha=0.8)
    ax_sr.plot(time_s, actual_bw / 1000, color=COLORS["True BW"], lw=1.5, ls="--", alpha=0.6)

    ax_sr.set_xlabel("Time (s)")
    ax_sr.set_ylabel("Sender Rate\n(Mbps)")
    ax_sr.set_ylim(bottom=0)
    ax_sr.grid(True, alpha=0.3)

    if zoom:
        ax_bw.set_xlim(zoom)
        ax_sr.set_xlim(zoom)

    plt.tight_layout()
    path = f"figures/{filename}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_overestimation_events(trace_type: TraceType, filename: str):
    """Plot overestimation (B̂ - B)+ over time to show when each algo over-injects."""
    config  = SimulationConfig(trace_type=trace_type)
    results = run_comparison(config, seed=SEED, verbose=False)

    stats0   = list(results.values())[0]["stats"]
    time_s   = np.array([s.time_ms / 1000.0 for s in stats0])

    fig, ax = plt.subplots(figsize=(10, 3))
    for algo in ["GCC", "NADA", "HPR (Proposed)"]:
        stats    = results[algo]["stats"]
        actual   = np.array([s.actual_bw_kbps for s in stats])
        estimate = np.array([s.estimated_bw_kbps for s in stats])
        overshoot = np.maximum(estimate - actual, 0) / np.maximum(actual, 1) * 100
        ax.plot(time_s, overshoot, color=COLORS[algo], lw=1.2, alpha=0.85, label=algo)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Overestimation %")
    ax.set_title(f"Overestimation Over Time — {trace_type.value.replace('_', ' ').title()}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = f"figures/{filename}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating time-series plots...")

    # Main traces for the paper
    plot_trace(TraceType.SUDDEN_DROP,  "ts_sudden_drop.png")
    plot_trace(TraceType.CELLULAR_4G,  "ts_cellular_4g.png")
    plot_trace(TraceType.WIFI_VARIABLE, "ts_wifi_variable.png")

    # Zoom into the first drop event (30–55 s) for sudden-drop
    plot_trace(TraceType.SUDDEN_DROP,  "ts_sudden_drop_zoom.png", zoom=(25, 60))

    # Overestimation event plots
    plot_overestimation_events(TraceType.SUDDEN_DROP,  "overest_sudden_drop.png")

    print("\nAll figures saved to figures/")
