"""
Robust location/scale: median and median absolute deviation (MAD), and the MAD-to-sigma
factor of a normal distribution.
"""

from typing import Any

import numpy as np

MAD_TO_SIGMA = 1.4826


def median_absolute_deviation(values: Any) -> tuple[float, float]:
    """``(median, MAD)`` of all elements of `values`; ``MAD_TO_SIGMA * MAD`` estimates sigma."""
    center = float(np.median(values))
    return center, float(np.median(np.abs(values - center)))
