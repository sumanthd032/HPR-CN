"""
HPR Bandwidth Estimation Simulator — Entry Point
=================================================
Runs all trace scenarios with multi-seed statistical variance,
ablation studies, and exports JSON results.

Usage
-----
    python main.py
"""

import json
import numpy as np

from hpr import (
    TraceType,
    SimulationConfig,
    run_comparison,
    run_ablation,
    run_multi_seed,
    export_results_json,
    ALGORITHM_KEYS,
    ABLATION_KEYS,
)

SEEDS = [42, 123, 456, 789, 1001]

TRACE_TYPES = [
    TraceType.STABLE,
    TraceType.FLUCTUATING,
    TraceType.SUDDEN_DROP,
    TraceType.GRADUAL_DECLINE,
    TraceType.CELLULAR_4G,
    TraceType.WIFI_VARIABLE,
]


def main() -> None:
    print("=" * 70)
    print("  HPR Bandwidth Estimation Simulator for Video Conferencing")
    print("=" * 70)

    # ================================================================
    # 1. Multi-seed comparison across all traces
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  MULTI-SEED COMPARISON  (seeds:", SEEDS, ")")
    print("=" * 70)

    all_multiseed: dict = {}
    for trace in TRACE_TYPES:
        print(f"\n\n{'#'*70}")
        print(f"  Trace: {trace.value.upper()}")
        print(f"{'#'*70}")
        config  = SimulationConfig(trace_type=trace)
        summary = run_multi_seed(config, seeds=SEEDS, verbose=True)
        all_multiseed[trace.value] = summary

    # ================================================================
    # 2. Print summary tables
    # ================================================================
    _print_mos_table(all_multiseed)
    _print_overest_table(all_multiseed)

    # ================================================================
    # 3. Ablation study on sudden-drop trace
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  ABLATION STUDY  (Sudden-Drop trace, seed 42)")
    print("=" * 70)
    abl_config   = SimulationConfig(trace_type=TraceType.SUDDEN_DROP)
    ablation_res = run_ablation(abl_config, seed=42, verbose=True)

    # Also run ablation on cellular trace
    print("\n  [Cellular 4G trace]")
    abl_config_4g  = SimulationConfig(trace_type=TraceType.CELLULAR_4G)
    ablation_4g    = run_ablation(abl_config_4g, seed=42, verbose=True)

    # ================================================================
    # 4. Detailed JSON exports
    # ================================================================
    # Detailed single-seed results for cellular trace
    config4g = SimulationConfig(trace_type=TraceType.CELLULAR_4G)
    np.random.seed(42)
    detailed = run_comparison(config4g, seed=42, verbose=False)
    export_results_json(detailed, "simulation_results.json")

    # Full multi-seed summary
    with open("all_results.json", "w", encoding="utf-8") as fh:
        json.dump(all_multiseed, fh, indent=2)
    print("\nMulti-seed results exported → all_results.json")

    # Ablation results
    ablation_export = {
        "sudden_drop": {k: v["metrics"] for k, v in ablation_res.items()},
        "cellular_4g": {k: v["metrics"] for k, v in ablation_4g.items()},
    }
    with open("ablation_results.json", "w", encoding="utf-8") as fh:
        json.dump(ablation_export, fh, indent=2)
    print("Ablation results exported → ablation_results.json")


# -----------------------------------------------------------------------
# Table helpers
# -----------------------------------------------------------------------

def _print_mos_table(all_results: dict) -> None:
    print("\n\n" + "=" * 80)
    print("  SUMMARY: Average MOS (mean ± std across 5 seeds, higher = better)")
    print("=" * 80)
    col = 18
    header = f"{'Trace':<22}"
    for key in ALGORITHM_KEYS:
        header += f" {key:<{col}}"
    print(header)
    print("-" * 80)
    for trace_name, algos in all_results.items():
        row = f"{trace_name:<22}"
        for key in ALGORITHM_KEYS:
            m = algos[key]["avg_mos"]["mean"]
            s = algos[key]["avg_mos"]["std"]
            cell = f"{m:.2f}±{s:.2f}"
            row += f" {cell:<{col}}"
        print(row)
    print()


def _print_overest_table(all_results: dict) -> None:
    print("\n" + "=" * 80)
    print("  OVERESTIMATION %  (mean ± std across 5 seeds, lower = better)")
    print("=" * 80)
    col = 18
    header = f"{'Trace':<22}"
    for key in ALGORITHM_KEYS:
        header += f" {key:<{col}}"
    print(header)
    print("-" * 80)
    for trace_name, algos in all_results.items():
        row = f"{trace_name:<22}"
        for key in ALGORITHM_KEYS:
            m = algos[key]["overestimation_pct"]["mean"]
            s = algos[key]["overestimation_pct"]["std"]
            cell = f"{m:.2f}±{s:.2f}"
            row += f" {cell:<{col}}"
        print(row)
    print()


if __name__ == "__main__":
    main()
