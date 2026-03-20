from __future__ import annotations

import math
from typing import Any, Optional


def read_ultrasonic_cm(motor: Any) -> Optional[float]:
    """
    Return ultrasonic distance in cm, or None for invalid/unavailable readings.
    """
    px = getattr(motor, "px", None)
    if px is None:
        return None

    try:
        d = float(px.get_distance())
    except Exception:
        return None

    if not math.isfinite(d) or d <= 0.0 or d > 500.0:
        return None
    return d


def ultrasonic_to_countdown(
    dist_cm: Optional[float],
    *,
    stop_cm: float = 20.0,
    caution_cm: float = 40.0,
) -> int:
    """
    Map the current front clearance into simple safety bands:
      d > caution_cm  -> 30
      stop_cm <= d <= caution_cm -> 10
      d < stop_cm  -> 1
    """
    if dist_cm is None:
        return 30

    d = float(dist_cm)
    if d < float(stop_cm):
        return 1
    if d <= float(caution_cm):
        return 10
    return 30