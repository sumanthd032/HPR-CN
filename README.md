# HPR: Hybrid Predictive-Reactive Bandwidth Estimation

> A real-time video conferencing bandwidth estimator that fuses delay-gradient
> feedback, linear trend prediction, and an adaptive Kalman filter.

---

## Overview

This repository accompanies the paper *"HPR: A Hybrid Predictive-Reactive
Algorithm for Bandwidth Estimation in Real-Time Video Conferencing"*.

Three algorithms are implemented and compared under six realistic network
scenarios:

| Algorithm | Type | Description |
|-----------|------|-------------|
| **AIMD** | Baseline | TCP-like additive increase / multiplicative decrease |
| **GCC** | Reactive | Delay-gradient state machine (RFC 8698-inspired) |
| **HPR** | **Proposed** | Reactive + predictive + adaptive Kalman fusion |

### HPR Novel Contributions

1. **Congestion-type classification** — distinguishes queue build-up, severe
   congestion, wireless random errors, and clear paths; each type triggers a
   differently-calibrated rate change rather than a single threshold.

2. **Adaptive Kalman process noise Q** — Q scales with the coefficient of
   variation (CV) of recent measurements.  Volatile channels get a larger Q
   (faster tracking); stable channels get a smaller Q (more smoothing).
   Prior work uses a fixed Q.

3. **Exponentially-weighted least-squares trend prediction** — recent
   measurement samples carry higher weight, making the slope estimate
   responsive to trend reversals without discarding historical context.

4. **Preemptive rate reduction** — when the trend predicts a future bandwidth
   decline and the channel is currently stable (low CV), HPR reduces rate
   *before* congestion is detected, eliminating the GCC reaction delay.

---

## Project Structure

```
HPR-CN/
├── hpr/                        # Core framework package
│   ├── __init__.py             # Public API
│   ├── traces.py               # Bandwidth trace generators (6 scenarios)
│   ├── network.py              # Packet-level network link simulator
│   ├── quality.py              # PSNR / MOS / FPS video quality model
│   ├── simulation.py           # SimulationConfig, run_simulation, run_comparison
│   ├── metrics.py              # Performance metrics (MAE, RMSE, MOS, convergence)
│   ├── export.py               # JSON serialisation for visualisation
│   └── estimators/
│       ├── __init__.py
│       ├── base.py             # BaseEstimator abstract class
│       ├── aimd.py             # AIMD baseline
│       ├── gcc.py              # GCC-inspired reactive estimator
│       └── hpr.py              # HPR proposed algorithm
├── main.py                     # CLI entry point
├── requirements.txt
├── simulation_results.json     # Cellular 4G detailed results (generated)
├── all_results.json            # All-trace summary results (generated)
├── HPR_IEEE_Paper.pdf          # Research paper (IEEE format)
└── HPR_Complete_Guide.docx     # Extended explanation and walkthrough
```

---

## Quick Start

```bash
# 1. Install dependencies (only numpy is strictly required)
pip install -r requirements.txt

# 2. Run all simulations (~15 s on a modern laptop)
python main.py
```

Output: per-algorithm metrics for every trace, a summary MOS table, and two
JSON result files.

---

## Usage as a Library

```python
from hpr import SimulationConfig, run_comparison, TraceType

config  = SimulationConfig(trace_type=TraceType.SUDDEN_DROP, duration_sec=120)
results = run_comparison(config, seed=42)

for algo, data in results.items():
    print(algo, data["metrics"]["avg_mos"])
```

### Custom estimator

```python
from hpr.estimators.base import BaseEstimator
from hpr.network import Packet
from typing import List

class MyEstimator(BaseEstimator):
    def update(self, packets: List[Packet], current_time_ms: float) -> float:
        # your logic here
        return self.estimate
```

### Add a new trace

Add a value to `TraceType` in `hpr/traces.py` and implement the bandwidth
pattern in `generate_bandwidth_trace()`.  The rest of the framework picks it
up automatically.

---

## Tuning HPR Parameters

Edit `HybridPredictiveEstimator` in `hpr/estimators/hpr.py` or pass kwargs
when instantiating:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `process_noise` (Q) | 500 | Base Kalman process noise — higher = faster adaptation |
| `measurement_noise` (R) | 120 | Kalman measurement noise — lower = trust measurements more |
| `prediction_weight` | 0.35 | Max weight of the predictive signal (0 = fully reactive) |
| `window_size` | 10 | Regression look-back window (samples) |
| `delay_threshold` | 10.0 ms | Overuse detection threshold |

---

## Network Trace Scenarios

| Scenario | Description | Typical use case |
|----------|-------------|-----------------|
| `STABLE` | 3 Mbps + Gaussian noise | Home broadband |
| `FLUCTUATING` | Sinusoidal 1.5–3.5 Mbps | Shared WiFi |
| `SUDDEN_DROP` | 4 → 1 Mbps drops at 30 s and 80 s | Background download starts |
| `GRADUAL_DECLINE` | 4 → 1.5 Mbps linear decay | Moving away from AP |
| `CELLULAR_4G` | 1.5–3 Mbps with bursty fades | Mobile on 4G/LTE |
| `WIFI_VARIABLE` | Interference + random drops | Office WiFi |

---

## Dependencies

- **Python ≥ 3.9**
- `numpy >= 1.24`
- `matplotlib >= 3.7` (only needed for figure generation)

---

## Citation

```
[Add citation after publication]
```

## License

Academic use. See your institution's guidelines.
