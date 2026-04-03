# HPR: Hybrid Predictive-Reactive Bandwidth Estimation

> A real-time video conferencing bandwidth estimator that fuses a five-class
> congestion classifier, exponentially-weighted trend prediction, and an
> adaptive Kalman filter — all in a closed control loop.

This repository accompanies the paper:

> **"HPR: A Hybrid Predictive-Reactive Algorithm for Bandwidth Estimation
> in Real-Time Video Conferencing"**
> B S Mahalakshmi, Sumantha, Siri R G, Suchand C Achar — BMS College of Engineering

---

## Background: Why Bandwidth Estimation is Hard

Every WebRTC video call sends packets at a rate controlled by a **bandwidth
estimator**. If the estimator guesses too high, the sender floods the network,
building queues and causing visible freezes. If it guesses too low, the video
quality suffers unnecessarily.

The difficulty is that the available bandwidth changes at three timescales
simultaneously:
- **Slow trends**: the user walks away from a WiFi AP, or transitions from
  4G to edge coverage.
- **Periodic fluctuations**: competing downloads cycle in and out on a shared link.
- **Abrupt drops**: a background upload suddenly starts.

Classical approaches react only *after* congestion is already measurable.
HPR adds a lightweight *predictive* layer so it can begin reducing its
estimate *before* the reactive signal fires.

---

## Algorithms Implemented

| Algorithm | Type | Description |
|-----------|------|-------------|
| **AIMD** | Baseline | Additive increase / multiplicative decrease on loss only |
| **GCC** | Reactive | Delay-gradient EWMA with overuse state machine (Holmer et al.) |
| **NADA** | Reactive | Unified delay+loss synthetic congestion signal (RFC 8698) |
| **HPR** | **Proposed** | Five-class classifier + trend predictor + adaptive Kalman |

> **Note:** RFC 8698 standardises **NADA**, *not* GCC. GCC is described
> in IETF draft-ietf-rmcat-gcc-02 (Holmer et al., 2016).

---

## HPR Architecture

```
  ┌──────────────────────────────────────────────────────────────┐
  │              Network Link (per 100 ms interval)              │
  │  Droptail queue · 50 ms prop. delay · Bernoulli loss p=1.5% │
  └──────────────────────┬───────────────────────────────────────┘
                         │ packets
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  LAYER 1 — REACTIVE                         [orange]         │
  │                                                              │
  │  OWD gradient  ──┐                                          │
  │                  ├──▶  5-Class Congestion Classifier  ──▶ B̂ᴿ │
  │  Loss rate ℓ  ───┘                                          │
  │                                                              │
  │  Classes: severe · delay · wireless · mild · clear           │
  │  Key novelty: "wireless" loss ≠ queue congestion             │
  └──────────────────────┬───────────────────────────────────────┘
                         │ B̂ history
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  LAYER 2 — PREDICTIVE                       [purple]         │
  │                                                              │
  │  Sliding window (n=10)  ──▶  Exp. Weighted OLS  ──▶  B̂ᴾ    │
  │                                                  └──▶  σ    │
  │  wᵢ = exp((i−1)/(n−1) − 1),  w₁≈0.37,  wₙ=1.0             │
  └──────────────────────┬───────────────────────────────────────┘
                         │ B̂ᴿ + B̂ᴾ + σ
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  LAYER 3 — ADAPTIVE KALMAN FUSION           [green]          │
  │                                                              │
  │  mₜ = (1−wσ)B̂ᴿ + wσB̂ᴾ          (w=0.35)                  │
  │  Qₜ = Q₀(1 + min(3·CVₜ, 3))     (Q₀=500, adaptive noise)   │
  │  B̂ₜ ← Kalman update(mₜ, Qₜ, R=120)                        │
  │  Preemptive cap if B̂ᴾ < 0.70·B̂ₜ₋₁ and σ > 0.5 and clear  │
  └──────────────────────┬───────────────────────────────────────┘
                         │ B̂ₜ (final estimate)
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  rₜ₊₁ = clip(0.9 × B̂ₜ,  100,  10 000 kbps)               │
  │  Sender rate set from ESTIMATOR, not ground-truth bandwidth  │
  └──────────────────────┬───────────────────────────────────────┘
                         │ (closed feedback loop ──────────────▶ Network Link)
```

See `figures/hpr_architecture.svg` for the full vector diagram.

---

## Five-Class Congestion Classifier

The most important novelty in HPR. Standard reactive controllers (GCC, NADA)
treat all packet loss the same way. HPR recognises that **wireless random errors**
do *not* mean the link is congested — the queue is not building.

| Class | Condition | Rate Response | Rationale |
|-------|-----------|---------------|-----------|
| **severe** | g̅ > θd AND ℓ > 4% | 0.78 × B̂ | Buffer overflow: aggressive back-off |
| **delay** | g̅ > θd (any ℓ) | 0.85 × B̂ | Queue building: drain buffer |
| **wireless** | ℓ > 4% AND g̅ ≤ θd | 0.92 × B̂ | Random errors, not queue: gentle cut |
| **mild** | 5 < g̅ ≤ θd OR mild loss | max(1.04B̂, B̂+80) | Cautious probe |
| **clear** | otherwise | max(1.15B̂, B̂+250) | Aggressive probe |

`θd = 10 ms` (standard GCC overuse threshold).

---

## Project Structure

```
HPR-CN/
├── hpr/                          # Core Python package
│   ├── __init__.py               # Public API (all exports)
│   ├── traces.py                 # Bandwidth trace generators (6 scenarios)
│   ├── network.py                # Packet-level network link simulator
│   ├── quality.py                # PSNR / MOS video quality proxy
│   ├── simulation.py             # Control loop, run_comparison, run_ablation,
│   │                             #   run_multi_seed
│   ├── metrics.py                # MAE, RMSE, MOS, overestimation%, convergence
│   ├── export.py                 # JSON serialisation of results
│   └── estimators/
│       ├── base.py               # BaseEstimator abstract class
│       ├── aimd.py               # AIMD baseline
│       ├── gcc.py                # GCC-inspired delay-gradient estimator
│       ├── nada.py               # NADA (RFC 8698) unified signal estimator
│       └── hpr.py                # HPR proposed algorithm (ablation_mode param)
│
├── main.py                       # Entry point: multi-seed comparison + ablation
├── plot_timeseries.py            # Generate time-series PNG figures
├── requirements.txt
│
├── figures/                      # Generated outputs
│   ├── hpr_architecture.svg      # Architecture diagram (hand-crafted SVG)
│   ├── ts_sudden_drop.png        # Time-series: all traces
│   ├── ts_sudden_drop_zoom.png   # Zoomed into first drop event (25–60 s)
│   ├── ts_cellular_4g.png
│   ├── ts_wifi_variable.png
│   └── overest_sudden_drop.png   # Overestimation (%) over time
│
├── simulation_results.json       # Cellular 4G detailed results (generated)
├── all_results.json              # Multi-seed summary all traces (generated)
└── ablation_results.json         # HPR ablation variants (generated)
```

---

## Quick Start

```bash
# 1. Install dependencies (numpy required; matplotlib for figures)
pip install -r requirements.txt

# 2. Run all simulations (5 seeds × 6 traces × 4 algorithms ≈ 60 s)
python main.py

# 3. Generate time-series plots
python plot_timeseries.py
```

**Outputs from `main.py`:**
- Per-algorithm metrics for every trace scenario
- MOS summary table (mean ± std, 5 seeds)
- Overestimation % table (mean ± std, 5 seeds)
- Multi-seed HPR ablation results (two traces)
- `all_results.json`, `ablation_results.json`, `simulation_results.json`

---

## Understanding the Simulation Loop

The simulation is a **closed control loop** — the estimator drives the sender:

```python
sending_rate = 1000   # kbps initial
for each 100 ms interval:
    packets = generate_packets(sending_rate)          # send at current rate
    transmit_through_link(packets)                    # queue, delay, loss
    B̂ = estimator.update(packets)                    # estimate from observations
    sending_rate = clip(0.9 * B̂, 100, 10_000)       # ← estimator output, NOT true BW
```

If the estimator overestimates (`B̂ > B_true`), the sender injects more
data than the link can carry → queue grows → delay and loss increase → the
estimator receives worse feedback. This feedback closes the loop and is what
makes the MOS, delay, and loss results differ between algorithms.

---

## Usage as a Library

```python
from hpr import SimulationConfig, run_comparison, run_ablation, TraceType

# Compare all four algorithms on a single trace
config  = SimulationConfig(trace_type=TraceType.SUDDEN_DROP, duration_sec=120)
results = run_comparison(config, seed=42)
for algo, data in results.items():
    print(algo, data["metrics"]["avg_mos"], data["metrics"]["overestimation_pct"])

# Multi-seed statistical comparison (returns mean ± std)
from hpr import run_multi_seed
summary = run_multi_seed(config, seeds=[42, 123, 456, 789, 1001])

# HPR ablation study
abl = run_ablation(config, seed=42)
# Variants: "HPR (Proposed)", "HPR-NoPred", "HPR-FixQ", "HPR-NoPreemp", "HPR-NoClassifier"
```

### HPR Ablation Modes

Pass `ablation_mode` to `HybridPredictiveEstimator`:

| Mode | What's disabled | Purpose |
|------|-----------------|---------|
| `None` (default) | — | Full HPR |
| `"no_predictor"` | Layer 2 (trend prediction) | Isolate classifier + Kalman |
| `"fixed_q"` | Adaptive Q scaling | Compare to static-Q Kalman |
| `"no_preemptive"` | Preemptive cap | Isolate preemptive benefit |
| `"no_classifier"` | 5-class → 2-state rule | Quantify classifier importance |

```python
from hpr.estimators.hpr import HybridPredictiveEstimator
est = HybridPredictiveEstimator(ablation_mode="no_predictor")
```

### Custom Estimator

```python
from hpr.estimators.base import BaseEstimator
from hpr.network import Packet
from typing import List

class MyEstimator(BaseEstimator):
    def update(self, packets: List[Packet], current_time_ms: float) -> float:
        # your bandwidth estimation logic
        return self.estimate
```

---

## Network Trace Scenarios

| Scenario | Description | BW Range | Real-World Analogue |
|----------|-------------|----------|---------------------|
| `STABLE` | 3 Mbps + Gaussian noise σ=50 | 2850–3150 kbps | Home broadband |
| `FLUCTUATING` | Sinusoidal load, 30 s period | 1500–3500 kbps | Shared WiFi, competing users |
| `SUDDEN_DROP` | 4 Mbps → 1 Mbps at 30 s, 80 s | 500–4000 kbps | Background download starts |
| `GRADUAL_DECLINE` | Linear 4 → 1.5 Mbps over 120 s | 1500–4000 kbps | Walking away from AP |
| `CELLULAR_4G` | Sine base + bursts + fades | 200–3500 kbps | 4G/LTE mobility |
| `WIFI_VARIABLE` | Dual-freq interference + drops | 1000–4500 kbps | Office WiFi, co-channel |

---

## HPR Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `process_noise` Q₀ | 500 (kbps)² | Base Kalman process noise |
| `measurement_noise` R | 120 (kbps)² | Kalman measurement noise |
| `prediction_weight` w | 0.35 | Max weight of predictive signal |
| `window_size` n | 10 | Regression window (intervals = 1 s) |
| `delay_threshold` θd | 10.0 ms | Overuse detection threshold |

The adaptive-Q mechanism (`Qₜ = Q₀ × (1 + min(3·CVₜ, 3))`) means these
parameters self-calibrate to local variability; they do not need per-network
re-tuning.

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| `mae_kbps` | Mean absolute error of estimate vs true BW |
| `rmse_kbps` | Root mean square error of estimate |
| `mape_percent` | Mean absolute percentage error |
| `utilization` | Mean sender rate / true bandwidth |
| `overestimation_pct` | Mean positive overshoot (→ queue build-up) |
| `avg_delay_ms` | Mean one-way packet delay |
| `avg_loss_rate` | Mean packet loss fraction |
| `avg_mos` | Mean Opinion Score (1–5, proxy via PSNR model) |
| `avg_convergence_ms` | Time to return within 20% of BW after a step change |

---

## Key Results (5 seeds, mean ± std)

**MOS** — sudden-drop trace: **HPR 3.20±0.02** vs GCC 3.10±0.10 vs NADA 2.68±0.03

**Overestimation %** — sudden-drop: **HPR 1.45±0.22** vs GCC 2.78±0.83 (−48%)

**Ablation** — removing the 5-class classifier ("HPR-NoClassifier") causes:
- RMSE +42% (1996 → 2831 kbps) on sudden-drop
- MOS −0.65 (3.20 → 2.55) on sudden-drop

---

## Limitations

- **Single flow only**: no multi-flow fairness, no cross-traffic coexistence.
- **IID Bernoulli loss**: real wireless links have burst correlation (Gilbert-Elliott); the wireless classifier may misfire under sustained bursts.
- **Simplified encoder**: instantaneous rate oracle, no encoder buffering or quantiser hysteresis.
- **Shared clock**: OWD measurement assumes synchronised sender/receiver clocks; production needs drift-robust alternatives.
- **No RMCAT testcases**: ns-3 RMCAT standard testcases are the required next step for deployment validation.

---

## Dependencies

```
numpy >= 1.24
matplotlib >= 3.7   # only for plot_timeseries.py
```

Install:
```bash
pip install -r requirements.txt
```

---

## License

Academic / research use only.
