"""calculate_reliability from the FRGB3D background model (Meng et al., J. Comput. Civ. Eng. 2027), unchanged."""
import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def calculate_reliability(ranges, reflectivity):
    """
    LiDAR point reliability model (Numba), for reflectivity over the full 0-100% range.
    Args:
    - ranges: range values (m)
    - reflectivity: reflectivity values (0-255)
    Returns:
    - position reliability (0-1)
    """
    # reliability array
    reliability = np.zeros_like(ranges, dtype=np.float32)
    min_std = 0.5

    # precomputed constants, extended to the full range
    a_low = (3-min_std) / (90.0 * 90.0)  # coefficient at 10% reflectivity
    a_high = (0.8-min_std) / (90.0 * 90.0)  # coefficient at 90% reflectivity
    refl_factor = (a_low - a_high) / (90.0 - 10.0)  # precomputed interpolation factor

    # coefficients at 0% and 100%
    a_at_0_percent = a_low + 10.0 * refl_factor
    a_at_100_percent = max(0.0, a_high - 10.0 * refl_factor)  # keep non-negative

    for i in prange(len(ranges)):  # parallel loop
        r = ranges[i]

        # reflectivity in percent
        refl = (reflectivity[i] / 255.0) * 100.0

        # coefficient from reflectivity (no clipping needed)
        if refl == 0.0:
            a = a_at_0_percent
        elif refl < 10.0:
            # linear interpolation over 0-10%
            t = refl / 10.0
            a = a_at_0_percent * (1.0 - t) + a_low * t
        elif refl == 10.0:
            # coefficient at 10% reflectivity
            a = a_low
        elif 10 < refl < 90.0:
            # original linear interpolation over 10-90%
            a = a_low - (refl - 10.0) * refl_factor
        elif refl == 90.0:
            # coefficient at 90% reflectivity
            a = a_high
        elif refl < 100.0:
            # linear interpolation over 90-100%
            t = (refl - 90.0) / 10.0
            a = a_high * (1.0 - t) + a_at_100_percent * t
        else:
            a = a_at_100_percent

        # quadratic standard deviation (cm)
        precision_std = a * (r * r) + min_std

        # reliability
        reliability[i] = max(0.0, 1.0 - 0.01 * precision_std)

    return reliability
