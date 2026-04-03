"""
Results Export
===============
Serialises simulation results to JSON for downstream visualisation
(e.g. a separate generate_figures.py script or a web dashboard).
"""

import json
from typing import Dict


def export_results_json(results: Dict, filepath: str, downsample: int = 240) -> None:
    """
    Write simulation results to a JSON file.

    Time-series data is down-sampled to at most *downsample* points to keep
    file sizes manageable for plotting without losing significant detail.

    Parameters
    ----------
    results    : dict  Output of ``run_comparison()``.
    filepath   : str   Destination JSON file path.
    downsample : int   Maximum number of time-series points to keep.
    """
    export_data: Dict = {}

    for algo_name, algo_data in results.items():
        stats  = algo_data["stats"]
        step   = max(1, len(stats) // downsample)
        sample = stats[::step]

        export_data[algo_name] = {
            "metrics":   algo_data["metrics"],
            "timeseries": {
                "time_sec":      [round(s.time_ms / 1000, 3) for s in sample],
                "actual_bw":     [round(s.actual_bw_kbps,    1) for s in sample],
                "estimated_bw":  [round(s.estimated_bw_kbps, 1) for s in sample],
                "sending_rate":  [round(s.sending_rate_kbps, 1) for s in sample],
                "delay_ms":      [round(s.delay_ms,          1) for s in sample],
                "loss_rate":     [round(s.loss_rate,         4) for s in sample],
            },
        }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(export_data, fh, indent=2)

    print(f"Results exported → {filepath}")
