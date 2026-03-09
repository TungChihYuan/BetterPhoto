"""
Stage 3 — White Balance
========================
Applies per-channel gain to neutralise the colour cast introduced by the
illuminant.  Two modes are supported:

  * Grey-world  (default) — assumes the scene average should be neutral grey.
  * Manual gains          — explicit R / G / B multipliers supplied by the user.
"""

import numpy as np


def white_balance(
    img: np.ndarray,
    r_gain: float | None = None,
    g_gain: float | None = None,
    b_gain: float | None = None,
) -> np.ndarray:
    """
    Apply white-balance gains to a linear RGB image.

    Parameters
    ----------
    img    : (H, W, 3) float32 linear RGB in [0, 1]
    r_gain : Red   gain (None → grey-world estimate)
    g_gain : Green gain (None → grey-world estimate)
    b_gain : Blue  gain (None → grey-world estimate)

    Returns
    -------
    (H, W, 3) float32 in [0, 1]
    """
    print("[3] White balance ...")

    if r_gain is None:
        # Grey-world: scale each channel so its mean equals the overall mean
        means  = img.mean(axis=(0, 1))          # shape (3,)
        grey   = means.mean()
        gains  = grey / (means + 1e-8)
    else:
        gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)

    return np.clip(img * gains[np.newaxis, np.newaxis, :], 0.0, 1.0)
