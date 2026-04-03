"""
Video Quality Estimation Model
================================
Maps sending rate and available bandwidth to perceptual quality metrics:
  - PSNR  (dB)       : Peak Signal-to-Noise Ratio proxy via H.264 rate–quality curve
  - MOS   (1–5)      : Mean Opinion Score derived from PSNR (ITU-T P.800-inspired)
  - FPS   (frames/s) : Effective frame rate tier based on bitrate

The overestimation penalty in PSNR models the video-quality degradation caused
by packet loss when the sender transmits faster than the bottleneck allows.
"""

import numpy as np


def estimate_video_quality(
    sending_rate_kbps: float,
    available_bw_kbps: float,
) -> dict:
    """
    Estimate video quality for a given operating point.

    Parameters
    ----------
    sending_rate_kbps : float  Actual encoding / sending rate.
    available_bw_kbps : float  True available bandwidth on the link.

    Returns
    -------
    dict with keys ``psnr`` (dB), ``mos`` (1–5), ``fps`` (int).
    """
    utilization = sending_rate_kbps / max(available_bw_kbps, 1.0)

    # ---- PSNR: logarithmic rate–quality model ----
    # Mirrors typical H.264 RD curves published in codec benchmarks.
    if sending_rate_kbps < 200:
        psnr = 25.0                                          # barely watchable
    else:
        psnr = 30.0 + 5.0 * np.log2(sending_rate_kbps / 500.0)

    # Overestimation penalty: excess traffic causes network congestion and
    # in-flight packet loss → visible artifacts (blocking, freezes)
    if utilization > 1.0:
        overuse = utilization - 1.0
        psnr   -= overuse * 15.0                            # heavy penalty

    psnr = float(np.clip(psnr, 20.0, 50.0))

    # ---- MOS: piecewise linear mapping (ITU-T P.800 inspired) ----
    if psnr >= 45.0:
        mos = 4.5
    elif psnr >= 38.0:
        mos = 3.5 + (psnr - 38.0) / 7.0
    elif psnr >= 30.0:
        mos = 2.5 + (psnr - 30.0) / 8.0
    else:
        mos = 1.0 + (psnr - 20.0) / 10.0 * 1.5

    mos = float(np.clip(mos, 1.0, 5.0))

    # ---- Effective frame rate: tier model ----
    if sending_rate_kbps > 1500:
        fps = 30
    elif sending_rate_kbps > 800:
        fps = 24
    elif sending_rate_kbps > 400:
        fps = 15
    else:
        fps = 10

    return {
        "psnr": round(psnr, 2),
        "mos":  round(mos,  2),
        "fps":  fps,
    }
