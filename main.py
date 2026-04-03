"""
HPR Bandwidth Estimation Simulator — Entry Point
=================================================
Runs all trace scenarios, prints per-algorithm metrics, exports a detailed
JSON result for the Cellular 4G trace, and prints a summary MOS table.

Usage
-----
    python main.py
"""

import numpy as np

from hpr import (
    TraceType,
    SimulationConfig,
    run_comparison,
    export_results_json,
    ALGORITHM_KEYS,
)


def main() -> None:
    print("=" * 62)
    print("  HPR Bandwidth Estimation Simulator for Video Conferencing")
    print("=" * 62)

    trace_types = [
        TraceType.STABLE,
        TraceType.FLUCTUATING,
        TraceType.SUDDEN_DROP,
        TraceType.GRADUAL_DECLINE,
        TraceType.CELLULAR_4G,
        TraceType.WIFI_VARIABLE,
    ]

    # ---- Run all traces, collect summary metrics ----
    all_metrics: dict = {}
    for trace in trace_types:
        print(f"\n\n{'#'*62}")
        print(f"  Trace: {trace.value.upper()}")
        print(f"{'#'*62}")
        config  = SimulationConfig(trace_type=trace)
        results = run_comparison(config)
        all_metrics[trace.value] = {
            name: data["metrics"] for name, data in results.items()
        }

    # ---- Detailed JSON export for the 4G trace ----
    config   = SimulationConfig(trace_type=TraceType.CELLULAR_4G)
    np.random.seed(42)
    detailed = run_comparison(config, seed=42)
    export_results_json(detailed, "simulation_results.json")

    # ---- Export full results for all traces ----
    export_all: dict = {}
    for trace in trace_types:
        config   = SimulationConfig(trace_type=trace)
        np.random.seed(42)
        results  = run_comparison(config, seed=42)
        export_all[trace.value] = {
            name: {"metrics": data["metrics"]} for name, data in results.items()
        }
    import json
    with open("all_results.json", "w", encoding="utf-8") as fh:
        json.dump(export_all, fh, indent=2)
    print("\nAll results exported → all_results.json")

    # ---- Summary table ----
    print("\n\n" + "=" * 72)
    print("  SUMMARY: Average MOS Score Across All Traces")
    print("=" * 72)
    col = 16
    header = f"{'Trace':<22}"
    for key in ALGORITHM_KEYS:
        header += f" {key:<{col}}"
    print(header)
    print("-" * 72)
    for trace_name, algos in all_metrics.items():
        row = f"{trace_name:<22}"
        for key in ALGORITHM_KEYS:
            mos = algos[key]["avg_mos"]
            row += f" {mos:<{col}}"
        print(row)
    print()


if __name__ == "__main__":
    main()
