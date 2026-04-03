"""
Simulation Engine
==================
Wires together the network link, a bandwidth estimator, and a sending-rate
controller for a complete simulation run.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from hpr.traces    import TraceType, generate_bandwidth_trace
from hpr.network   import NetworkLink, NetworkStats, Packet
from hpr.estimators import AIMDEstimator, GCCEstimator, NADAEstimator, HybridPredictiveEstimator
from hpr.estimators.base import BaseEstimator
from hpr.metrics   import compute_metrics


# ---- Configuration --------------------------------------------------------

@dataclass
class SimulationConfig:
    """All tuneable parameters for a single simulation run."""
    trace_type:         TraceType = TraceType.CELLULAR_4G
    duration_sec:       int       = 120
    interval_ms:        int       = 100
    packet_size_bytes:  int       = 1200
    base_delay_ms:      float     = 50.0
    jitter_ms:          float     = 10.0
    queue_size:         int       = 50
    initial_rate_kbps:  float     = 1000.0
    # Safety margin: sender uses 90 % of estimated bandwidth
    sending_margin:     float     = 0.90


# ---- Single simulation run ------------------------------------------------

def run_simulation(
    config:    SimulationConfig,
    estimator: BaseEstimator,
) -> List[NetworkStats]:
    """
    Run a complete simulation for *config.duration_sec* seconds.

    At each time step (every ``interval_ms`` ms):
      1. Generate packets at the current sending rate.
      2. Pass packets through the network link.
      3. Feed packet observations to the estimator.
      4. Adapt the sending rate to ``estimate × sending_margin``.

    Parameters
    ----------
    config    : SimulationConfig  Simulation parameters.
    estimator : BaseEstimator     The algorithm under test.

    Returns
    -------
    List[NetworkStats]  One entry per time step.
    """
    bw_trace = generate_bandwidth_trace(
        config.trace_type, config.duration_sec, config.interval_ms
    )
    link = NetworkLink(
        bandwidth_trace    = bw_trace,
        interval_ms        = config.interval_ms,
        base_delay_ms      = config.base_delay_ms,
        jitter_ms          = config.jitter_ms,
        queue_size_packets = config.queue_size,
    )

    sending_rate = config.initial_rate_kbps
    stats: List[NetworkStats] = []

    for step in range(len(bw_trace)):
        current_time_ms = step * config.interval_ms
        actual_bw       = link.get_current_bandwidth()

        # ---- Generate packets for this interval ----
        bytes_per_interval = (sending_rate * config.interval_ms) / 8.0
        n_packets = max(1, int(bytes_per_interval / config.packet_size_bytes))

        packets: List[Packet] = []
        for i in range(n_packets):
            send_offset = i * (config.interval_ms / n_packets)
            pkt = Packet(
                seq_num       = step * 100 + i,
                size_bytes    = config.packet_size_bytes,
                send_time_ms  = current_time_ms + send_offset,
            )
            link.send_packet(pkt)
            packets.append(pkt)

        # ---- Collect stats for THIS interval (packets sent at current sending_rate) ----
        received      = [p for p in packets if not p.lost]
        avg_delay     = (float(np.mean([p.recv_time_ms - p.send_time_ms
                                        for p in received]))
                         if received else 0.0)
        loss_rate     = sum(1 for p in packets if p.lost) / len(packets)
        effective_rate = sending_rate   # rate that WAS used to generate this interval's packets

        # ---- Update estimator ----
        estimated_bw = estimator.update(packets, current_time_ms)

        # ---- Adapt sending rate for NEXT interval ----
        # The estimator output B̂_t directly sets the next sender bitrate:
        #   r_{t+1} = 0.9 × B̂_t
        # This is the closed-loop property: the estimator drives the sender,
        # NOT the ground-truth bandwidth.
        sending_rate = float(np.clip(
            estimated_bw * config.sending_margin, 100, 10_000
        ))

        stats.append(NetworkStats(
            time_ms           = current_time_ms,
            actual_bw_kbps    = actual_bw,
            estimated_bw_kbps = estimated_bw,
            sending_rate_kbps = effective_rate,   # rate actually used this interval
            delay_ms          = avg_delay,
            loss_rate         = loss_rate,
            queue_size        = len(link.queue),
        ))

        link.step()

    return stats


# ---- Multi-algorithm comparison run --------------------------------------

#: Canonical names used as dict keys throughout the codebase.
ALGORITHM_KEYS = ["AIMD", "GCC", "NADA", "HPR (Proposed)"]

#: Ablation variant keys.
ABLATION_KEYS = [
    "HPR (Proposed)",
    "HPR-NoPred",
    "HPR-FixQ",
    "HPR-NoPreemp",
    "HPR-NoClassifier",
]


def run_comparison(config: SimulationConfig, seed: int = 42, verbose: bool = True) -> dict:
    """
    Run AIMD, GCC, NADA, and HPR on the *same* bandwidth trace.

    Using a fixed ``seed`` ensures all algorithms see identical network
    conditions — the only variable is the estimator, making the comparison fair.

    Returns
    -------
    dict  Keyed by algorithm name (see ``ALGORITHM_KEYS``).
          Each value: ``{"stats": List[NetworkStats], "metrics": dict}``.
    """
    algorithms: dict = {
        "AIMD":            AIMDEstimator(config.initial_rate_kbps),
        "GCC":             GCCEstimator(config.initial_rate_kbps),
        "NADA":            NADAEstimator(config.initial_rate_kbps),
        "HPR (Proposed)":  HybridPredictiveEstimator(config.initial_rate_kbps),
    }

    results: dict = {}
    for name, estimator in algorithms.items():
        np.random.seed(seed)
        stats   = run_simulation(config, estimator)
        metrics = compute_metrics(stats)
        results[name] = {"stats": stats, "metrics": metrics}

        if verbose:
            print(f"\n{'='*50}")
            print(f"Algorithm : {name}")
            print(f"{'='*50}")
            for key, val in metrics.items():
                print(f"  {key}: {val}")

    return results


def run_ablation(config: SimulationConfig, seed: int = 42, verbose: bool = True) -> dict:
    """
    Run HPR ablation variants to isolate the contribution of each component.

    Variants
    --------
    ``HPR (Proposed)``    — full HPR (all three layers)
    ``HPR-NoPred``        — Layer 2 disabled (reactive + adaptive-Q Kalman only)
    ``HPR-FixQ``          — Layer 3 uses fixed Q = Q0 (no adaptive scaling)
    ``HPR-NoPreemp``      — preemptive reduction step disabled
    ``HPR-NoClassifier``  — Layer 1 uses a simple 2-state classifier

    Returns
    -------
    dict  Keyed by variant name. Each value: ``{"stats": ..., "metrics": ...}``.
    """
    variants: dict = {
        "HPR (Proposed)":   HybridPredictiveEstimator(config.initial_rate_kbps),
        "HPR-NoPred":       HybridPredictiveEstimator(config.initial_rate_kbps, ablation_mode="no_predictor"),
        "HPR-FixQ":         HybridPredictiveEstimator(config.initial_rate_kbps, ablation_mode="fixed_q"),
        "HPR-NoPreemp":     HybridPredictiveEstimator(config.initial_rate_kbps, ablation_mode="no_preemptive"),
        "HPR-NoClassifier": HybridPredictiveEstimator(config.initial_rate_kbps, ablation_mode="no_classifier"),
    }

    results: dict = {}
    for name, estimator in variants.items():
        np.random.seed(seed)
        stats   = run_simulation(config, estimator)
        metrics = compute_metrics(stats)
        results[name] = {"stats": stats, "metrics": metrics}

        if verbose:
            print(f"\n  {name}: overest={metrics['overestimation_pct']:.2f}%"
                  f"  delay={metrics['avg_delay_ms']:.0f}ms"
                  f"  MOS={metrics['avg_mos']:.2f}"
                  f"  RMSE={metrics['rmse_kbps']:.0f}kbps")

    return results


def run_multi_seed(
    config: SimulationConfig,
    seeds: list = None,
    verbose: bool = False,
) -> dict:
    """
    Run all algorithms over multiple random seeds and return per-metric
    mean ± std, enabling statistical variance reporting.

    Returns
    -------
    dict  Two-level: outer key = algorithm, inner key = metric name.
          Each leaf: ``{"mean": float, "std": float}``.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1001]

    # Accumulate per-algorithm, per-metric lists
    accumulator: dict = {key: {} for key in ALGORITHM_KEYS}

    for seed in seeds:
        results = run_comparison(config, seed=seed, verbose=False)
        for algo, data in results.items():
            for metric, value in data["metrics"].items():
                accumulator[algo].setdefault(metric, []).append(value)

    # Compute mean ± std
    summary: dict = {}
    for algo, metric_lists in accumulator.items():
        summary[algo] = {}
        for metric, values in metric_lists.items():
            arr = np.array(values)
            summary[algo][metric] = {
                "mean": round(float(np.mean(arr)), 4),
                "std":  round(float(np.std(arr)),  4),
            }

    if verbose:
        for algo, metrics in summary.items():
            print(f"\n{algo}:")
            for k, v in metrics.items():
                print(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")

    return summary
