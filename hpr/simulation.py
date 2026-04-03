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
from hpr.estimators import AIMDEstimator, GCCEstimator, HybridPredictiveEstimator
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

        # ---- Update estimator ----
        estimated_bw = estimator.update(packets, current_time_ms)

        # ---- Adapt sending rate ----
        sending_rate = float(np.clip(
            estimated_bw * config.sending_margin, 100, 10_000
        ))

        # ---- Collect stats ----
        received  = [p for p in packets if not p.lost]
        avg_delay = (float(np.mean([p.recv_time_ms - p.send_time_ms
                                    for p in received]))
                     if received else 0.0)
        loss_rate = sum(1 for p in packets if p.lost) / len(packets)

        stats.append(NetworkStats(
            time_ms           = current_time_ms,
            actual_bw_kbps    = actual_bw,
            estimated_bw_kbps = estimated_bw,
            sending_rate_kbps = sending_rate,
            delay_ms          = avg_delay,
            loss_rate         = loss_rate,
            queue_size        = len(link.queue),
        ))

        link.step()

    return stats


# ---- Multi-algorithm comparison run --------------------------------------

#: Canonical names used as dict keys throughout the codebase.
ALGORITHM_KEYS = ["AIMD", "GCC", "HPR (Proposed)"]


def run_comparison(config: SimulationConfig, seed: int = 42) -> dict:
    """
    Run AIMD, GCC, and HPR (Proposed) on the *same* bandwidth trace.

    Using a fixed ``seed`` ensures all three algorithms see identical network
    conditions — the only variable is the estimator, making the comparison fair.

    Returns
    -------
    dict  Keyed by algorithm name (see ``ALGORITHM_KEYS``).
          Each value: ``{"stats": List[NetworkStats], "metrics": dict}``.
    """
    algorithms: dict = {
        "AIMD":          AIMDEstimator(config.initial_rate_kbps),
        "GCC":           GCCEstimator(config.initial_rate_kbps),
        "HPR (Proposed)": HybridPredictiveEstimator(config.initial_rate_kbps),
    }

    results: dict = {}
    for name, estimator in algorithms.items():
        np.random.seed(seed)                       # identical trace for every algo
        stats   = run_simulation(config, estimator)
        metrics = compute_metrics(stats)
        results[name] = {"stats": stats, "metrics": metrics}

        print(f"\n{'='*50}")
        print(f"Algorithm : {name}")
        print(f"{'='*50}")
        for key, val in metrics.items():
            print(f"  {key}: {val}")

    return results
