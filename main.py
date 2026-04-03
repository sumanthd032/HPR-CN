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
    # 3. Multi-seed ablation study
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  MULTI-SEED ABLATION STUDY  (Sudden-Drop trace, 5 seeds)")
    print("=" * 70)
    abl_config    = SimulationConfig(trace_type=TraceType.SUDDEN_DROP)
    abl_multiseed = _run_ablation_multiseed(abl_config, SEEDS)

    print("\n  [Cellular 4G trace, 5 seeds]")
    abl_config_4g  = SimulationConfig(trace_type=TraceType.CELLULAR_4G)
    abl_multi_4g   = _run_ablation_multiseed(abl_config_4g, SEEDS)

    # Single-seed ablation for the tables
    ablation_res = run_ablation(abl_config, seed=42, verbose=False)
    ablation_4g  = run_ablation(abl_config_4g, seed=42, verbose=False)

    # ================================================================
    # 4. Detailed JSON exports
    # ================================================================
    # Print multi-seed ablation tables
    _print_ablation_table("Sudden-Drop", abl_multiseed)
    _print_ablation_table("Cellular-4G", abl_multi_4g)
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
        "sudden_drop_seed42":     {k: v["metrics"] for k, v in ablation_res.items()},
        "cellular_4g_seed42":     {k: v["metrics"] for k, v in ablation_4g.items()},
        "sudden_drop_multiseed":  abl_multiseed,
        "cellular_4g_multiseed":  abl_multi_4g,
    }
    with open("ablation_results.json", "w", encoding="utf-8") as fh:
        json.dump(ablation_export, fh, indent=2)
    print("Ablation results exported → ablation_results.json")


# -----------------------------------------------------------------------
# Table helpers
# -----------------------------------------------------------------------

def _run_ablation_multiseed(config, seeds):
    """Run ablation variants over multiple seeds; return mean±std per metric."""
    from hpr import run_ablation, ABLATION_KEYS
    accumulator = {k: {} for k in ABLATION_KEYS}
    for seed in seeds:
        results = run_ablation(config, seed=seed, verbose=False)
        for variant, data in results.items():
            for metric, value in data["metrics"].items():
                accumulator[variant].setdefault(metric, []).append(value)
    summary = {}
    for variant, metric_lists in accumulator.items():
        summary[variant] = {}
        for metric, values in metric_lists.items():
            arr = np.array(values)
            summary[variant][metric] = {
                "mean": round(float(np.mean(arr)), 4),
                "std":  round(float(np.std(arr)),  4),
            }
    return summary


def _print_ablation_table(trace_name, ablation_summary):
    print(f"\n  Ablation ({trace_name}, mean±std over 5 seeds):")
    fmt = "    {:<22} overest={:>6.2f}±{:.2f}%  delay={:>5.0f}±{:.0f}ms  MOS={:.2f}±{:.2f}  RMSE={:.0f}±{:.0f}"
    for variant, metrics in ablation_summary.items():
        print(fmt.format(
            variant,
            metrics["overestimation_pct"]["mean"], metrics["overestimation_pct"]["std"],
            metrics["avg_delay_ms"]["mean"],        metrics["avg_delay_ms"]["std"],
            metrics["avg_mos"]["mean"],              metrics["avg_mos"]["std"],
            metrics["rmse_kbps"]["mean"],            metrics["rmse_kbps"]["std"],
        ))


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
