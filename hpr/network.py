"""
Packet-Level Network Link Simulator
=====================================
Models a single bottleneck link with:
  - Variable bandwidth driven by a pre-generated trace
  - Propagation delay + Gaussian jitter
  - Droptail queue with configurable depth
  - Wireless random-error loss (~1.5 %)

This is intentionally simple: it captures the key dynamics that bandwidth
estimators must react to without modelling full protocol stacks.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Packet:
    """A single network packet (or RTP-sized data unit)."""
    seq_num:       int
    size_bytes:    int
    send_time_ms:  float
    recv_time_ms:  float  = 0.0
    lost:          bool   = False


@dataclass
class NetworkStats:
    """Snapshot of network state at one simulation time step."""
    time_ms:            float
    actual_bw_kbps:     float
    estimated_bw_kbps:  float
    sending_rate_kbps:  float
    delay_ms:           float
    loss_rate:          float
    queue_size:         int


class NetworkLink:
    """
    Simulates a bottleneck link driven by a pre-computed bandwidth trace.

    The model captures four delay components that matter for estimators:
      1. Transmission delay  — size / bandwidth (varies with trace)
      2. Propagation delay   — fixed RTT/2 base + Gaussian jitter
      3. Queuing delay       — proportional to bytes ahead in FIFO queue
      4. Random loss         — independent Bernoulli drops (wireless errors)

    Droptail: when the queue is full, arriving packets are dropped.
    """

    WIRELESS_LOSS_PROB = 0.015   # ~1.5 % random bit-error loss

    def __init__(
        self,
        bandwidth_trace:       np.ndarray,
        interval_ms:           int   = 100,
        base_delay_ms:         float = 50.0,
        jitter_ms:             float = 10.0,
        queue_size_packets:    int   = 50,
    ) -> None:
        self.bw_trace    = bandwidth_trace
        self.interval_ms = interval_ms
        self.base_delay  = base_delay_ms
        self.jitter      = jitter_ms
        self.max_queue   = queue_size_packets

        self.current_step: int          = 0
        self.queue:        List[Packet] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_bandwidth(self) -> float:
        """Return the bandwidth (kbps) at the current simulation step."""
        idx = min(self.current_step, len(self.bw_trace) - 1)
        return float(self.bw_trace[idx])

    def send_packet(self, packet: Packet) -> Packet:
        """
        Attempt to transmit *packet* through the link.

        The packet is marked lost if:
          a) the droptail queue is full, or
          b) a random wireless error occurs.

        Otherwise ``recv_time_ms`` is set and the packet joins the queue.
        """
        bw_kbps = self.get_current_bandwidth()

        # --- Droptail check ---
        if len(self.queue) >= self.max_queue:
            packet.lost = True
            return packet

        # --- Random wireless error ---
        if np.random.random() < self.WIRELESS_LOSS_PROB:
            packet.lost = True
            return packet

        # --- Compute receive time ---
        transmission_delay_ms = (packet.size_bytes * 8) / bw_kbps   # ms
        prop_delay_ms = max(5.0, self.base_delay + np.random.normal(0, self.jitter))

        # Queuing delay: sum of transmission times for bytes already queued
        queued_bytes = sum(p.size_bytes for p in self.queue)
        queuing_delay_ms = (queued_bytes * 8) / bw_kbps              # ms

        packet.recv_time_ms = (packet.send_time_ms
                               + transmission_delay_ms
                               + prop_delay_ms
                               + queuing_delay_ms)
        self.queue.append(packet)
        return packet

    def step(self) -> None:
        """Advance one time step and drain delivered packets from the queue."""
        self.current_step += 1
        current_time_ms = self.current_step * self.interval_ms
        self.queue = [p for p in self.queue if p.recv_time_ms > current_time_ms]
