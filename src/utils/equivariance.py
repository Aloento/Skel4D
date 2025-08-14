"""
Equivariance utilities for building invertible transforms (wrapping SK implementation).

Uses StableKeypoints' RandomAffineWithInverse to apply forward/inverse warps.
"""
from typing import Tuple

from StableKeypoints.data.transforms import RandomAffineWithInverse


def build_random_invertible_transform(
    degrees: float = 30.0,
    scale: Tuple[float, float] = (0.9, 1.1),
    translate: Tuple[float, float] = (0.1, 0.1),
) -> RandomAffineWithInverse:
    """
    Factory to create a StableKeypoints-style invertible affine transform.

    Args:
        degrees: Max absolute rotation in degrees (uniform in [-d, d]).
        scale: (min, max) isotropic scale range.
        translate: (tx, ty) as fractions of image size in [-t, t].

    Returns:
        RandomAffineWithInverse instance with the provided ranges.
    """
    return RandomAffineWithInverse(
        degrees=degrees,
        scale=scale,
        translate=translate,
    )
