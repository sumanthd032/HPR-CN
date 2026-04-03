"""
HPR — Hybrid Predictive-Reactive Bandwidth Estimation Framework
================================================================
Public API for the simulation framework.

Typical usage
-------------
>>> from hpr import SimulationConfig, run_comparison, TraceType
>>> config  = SimulationConfig(trace_type=TraceType.CELLULAR_4G)
>>> results = run_comparison(config, seed=42)
"""

from hpr.traces      import TraceType, generate_bandwidth_trace
from hpr.network     import Packet, NetworkStats, NetworkLink
from hpr.quality     import estimate_video_quality
from hpr.metrics     import compute_metrics
from hpr.export      import export_results_json
from hpr.simulation  import SimulationConfig, run_simulation, run_comparison, ALGORITHM_KEYS
from hpr.estimators  import (
    BaseEstimator,
    AIMDEstimator,
    GCCEstimator,
    HybridPredictiveEstimator,
)

__all__ = [
    # Traces
    "TraceType",
    "generate_bandwidth_trace",
    # Network
    "Packet",
    "NetworkStats",
    "NetworkLink",
    # Quality
    "estimate_video_quality",
    # Metrics
    "compute_metrics",
    # Export
    "export_results_json",
    # Simulation
    "SimulationConfig",
    "run_simulation",
    "run_comparison",
    "ALGORITHM_KEYS",
    # Estimators
    "BaseEstimator",
    "AIMDEstimator",
    "GCCEstimator",
    "HybridPredictiveEstimator",
]
