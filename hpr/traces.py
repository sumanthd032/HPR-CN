"""
Network Bandwidth Trace Generator
===================================
Generates synthetic but realistic bandwidth traces (kbps) modelling
common real-world scenarios encountered in video conferencing:
stable home broadband, shared WiFi, 4G/LTE mobility, etc.
"""

import numpy as np
from enum import Enum


class TraceType(Enum):
    STABLE           = "stable"
    FLUCTUATING      = "fluctuating"
    SUDDEN_DROP      = "sudden_drop"
    GRADUAL_DECLINE  = "gradual_decline"
    CELLULAR_4G      = "cellular_4g"
    WIFI_VARIABLE    = "wifi_variable"


def generate_bandwidth_trace(
    trace_type: TraceType,
    duration_sec: int = 120,
    interval_ms: int = 100,
) -> np.ndarray:
    """
    Generate a bandwidth trace sampled every ``interval_ms`` ms.

    All traces are clipped to [100, 10 000] kbps to stay within realistic
    range for real-time video conferencing (100 kbps floor = audio-only
    fallback; 10 Mbps ceiling = high-quality 4K stream).

    Parameters
    ----------
    trace_type  : TraceType   Which scenario to simulate.
    duration_sec: int         Total trace length in seconds (default 120).
    interval_ms : int         Sampling resolution in ms (default 100).

    Returns
    -------
    np.ndarray  Bandwidth values at each time step (kbps).
    """
    n_steps = (duration_sec * 1000) // interval_ms
    t = np.linspace(0, duration_sec, n_steps)

    if trace_type == TraceType.STABLE:
        # Stable ~3 Mbps home broadband with minor thermal noise
        bw = 3000 + np.random.normal(0, 50, n_steps)

    elif trace_type == TraceType.FLUCTUATING:
        # Shared WiFi: sinusoidal load from competing users every ~30 s
        bw = (2500
              + 1000 * np.sin(2 * np.pi * t / 30)
              + np.random.normal(0, 100, n_steps))

    elif trace_type == TraceType.SUDDEN_DROP:
        # Sudden congestion events (e.g. background download starts)
        # Drop 1: 30–50 s  →  1 000 kbps
        # Drop 2: 80–90 s  →    500 kbps
        bw = np.full(n_steps, 4000.0)
        drop_events = [
            (30, 50,  1000.0),
            (80, 90,   500.0),
        ]
        for start_sec, end_sec, level in drop_events:
            i0 = min(int(start_sec * 1000 / interval_ms), n_steps - 1)
            i1 = min(int(end_sec   * 1000 / interval_ms), n_steps)
            bw[i0:i1] = level
        bw += np.random.normal(0, 80, n_steps)

    elif trace_type == TraceType.GRADUAL_DECLINE:
        # Moving away from a WiFi access point: 4 Mbps → 1.5 Mbps over 120 s
        bw = np.linspace(4000, 1500, n_steps) + np.random.normal(0, 60, n_steps)

    elif trace_type == TraceType.CELLULAR_4G:
        # Realistic 4G/LTE: slow sinusoidal base, exponential micro-bursts,
        # and occasional *sustained* deep fades (not instantaneous drops).
        base   = 2000 + 500 * np.sin(2 * np.pi * t / 45)
        bursts = np.random.exponential(300, n_steps)

        # Sustained fades: find rare onset points, apply smooth ramp
        fades = np.zeros(n_steps)
        onset_mask = np.random.random(n_steps) < 0.005   # ~1 fade per 20 s
        for fs in np.where(onset_mask)[0]:
            fade_len = np.random.randint(10, 30)          # 1–3 s at 100 ms res
            fe = min(fs + fade_len, n_steps)
            half = max((fe - fs) // 2, 1)
            for k in range(fs, fe):
                # Triangular: deepest at midpoint, tapers at edges
                depth = -800 * (1.0 - abs(k - (fs + half)) / half)
                fades[k] = min(fades[k], depth)           # keep deepest if overlap

        bw = base + bursts + fades

    elif trace_type == TraceType.WIFI_VARIABLE:
        # WiFi with dual-frequency interference (channel congestion)
        # plus occasional association re-negotiation drops
        base          = 3500
        interference  = (800
                         * np.sin(2 * np.pi * t / 20)
                         * np.sin(2 * np.pi * t / 7))
        random_drops  = np.where(np.random.random(n_steps) < 0.03, -1500, 0)
        bw = base + interference + random_drops + np.random.normal(0, 100, n_steps)

    else:
        raise ValueError(f"Unknown trace type: {trace_type}")

    return np.clip(bw, 100, 10_000)
