"""
Base class for all bandwidth estimators.
"""

from typing import List
from hpr.network import Packet


class BaseEstimator:
    """
    Abstract base for all bandwidth estimators.

    Subclasses implement ``update()``, which receives the list of packets
    observed during the current time step and returns the new bandwidth
    estimate in kbps.
    """

    def __init__(self, initial_estimate_kbps: float = 1000.0) -> None:
        self.estimate: float        = initial_estimate_kbps
        self.history:  List[float]  = [initial_estimate_kbps]

    def update(self, packets: List[Packet], current_time_ms: float) -> float:
        raise NotImplementedError

    def get_estimate(self) -> float:
        return self.estimate
