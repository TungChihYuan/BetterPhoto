"""
Stage 6 — Tone Mapping
=======================
Compresses a linear HDR-like image into the display range [0, 1].

Available operators
-------------------
reinhard  — Global Reinhard (2002):   L_out = L / (1 + L)
filmic    — Hejl-Burgess-Dawson curve, widely used in games/VFX
"""

import numpy as np


def tone_mapping(img: np.ndarray, method: str = "reinhard") -> np.ndarray:
    """
    Apply a tone-mapping operator to a linear RGB image.

    Parameters
    ----------
    img    : (H, W, 3) float32 linear RGB  (may contain values > 1)
    method : 'reinhard' | 'filmic'

    Returns
    -------
    (H, W, 3) float32 in [0, 1]
    """
    print(f"[6] Tone mapping ({method}) ...")

    if method == "reinhard":
        # L_out = L / (1 + L)
        return img / (1.0 + img)

    elif method == "filmic":
        # Hejl-Burgess-Dawson approximation
        # Ref: http://filmicworlds.com/blog/filmic-tonemapping-operators/
        x = np.maximum(img - 0.004, 0.0)
        return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)

    else:
        raise ValueError(f"Unknown tone-mapping method: '{method}'. "
                         "Choose 'reinhard' or 'filmic'.")
