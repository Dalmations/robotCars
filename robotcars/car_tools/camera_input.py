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


def ultrasonic_to_countdown(dist_cm: Optional[float]) -> int:
    """
    Map the current front clearance into simple safety bands:
      d > 40cm  -> 30
      20cm <= d <= 40cm -> 10
      d < 20cm  -> 1
    """
    if dist_cm is None:
        return 30

    d = float(dist_cm)
    if d < 20.0:
        return 1
    if d <= 40.0:
        return 10
    return 30
