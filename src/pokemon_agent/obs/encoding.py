"""In-place numerical encodings shared by the observation encoders."""

from __future__ import annotations

from typing import Any


def two_hot_encode_inplace(
    value: float, 
    n_bins: int, 
    buffer: Any, 
    start_idx: int, 
    v_min: float = 0.0, 
    v_max: float = 1.0
) -> None:
    """
    Applies two-hot encoding to represent a continuous value across discrete bins.
    
    Instead of assigning a continuous scalar to a single bucket (which loses 
    gradient information), this splits the weight (1.0) proportionally across 
    the two closest adjacent bins. 

    Args:
        value: The continuous scalar to encode.
        n_bins: The total number of discrete bins available.
        buffer: The shared NumPy array being written to.
        start_idx: The starting index for this feature block in the buffer.
        v_min: The minimum expected value (clamping boundary).
        v_max: The maximum expected value (clamping boundary).
    """
    # 1. Clamp value to the expected range
    val = max(v_min, min(v_max, float(value)))
    
    # 2. Scale the value to the bin index space [0, n_bins - 1]
    scaled = (val - v_min) / (v_max - v_min) * (n_bins - 1)
    
    # 3. Calculate the lower bin index and the fractional remainder
    idx = int(scaled)
    frac = scaled - float(idx)
    
    
    
    # 4. Write weights in-place
    if idx < n_bins - 1:
        # Distribute 1.0 across the lower and upper bin based on proximity
        buffer[start_idx + idx] = 1.0 - frac
        buffer[start_idx + idx + 1] = frac
    else:
        # Edge case: exactly at or exceeding the maximum boundary
        buffer[start_idx + n_bins - 1] = 1.0