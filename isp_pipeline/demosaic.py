"""
Stage 2 — Bayer Demosaicing
============================
Bilinear interpolation on RGGB / BGGR / GRBG / GBRG Bayer patterns.

Algorithm
---------
Each colour channel is placed at its known mosaic positions and then
interpolated to every pixel using a weighted bilinear kernel via
`scipy.ndimage.convolve`.  No external imaging libraries are required.
"""

import numpy as np
from scipy.ndimage import convolve


# ──────────────────────────────────────────────────────────────────────────────
# Kernels
# ──────────────────────────────────────────────────────────────────────────────

# Used for R and B channels (average from 4 diagonal or cross neighbours)
_KERNEL_RB = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
], dtype=np.float32) / 4.0

# Used for the G channel (cross-shaped average)
_KERNEL_G = np.array([
    [0, 1, 0],
    [1, 4, 1],
    [0, 1, 0],
], dtype=np.float32) / 4.0


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def bayer_demosaic(bayer: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Demosaic a Bayer CFA image into a full-colour RGB image.

    Parameters
    ----------
    bayer   : (H, W) float32 in [0, 1]
    pattern : one of 'RGGB', 'BGGR', 'GRBG', 'GBRG'

    Returns
    -------
    rgb : (H, W, 3) float32 in [0, 1]   — channel order R, G, B
    """
    print(f"[2] Bayer demosaicing (pattern={pattern}) ...")
    pattern = pattern.upper()
    if len(pattern) != 4 or set(pattern) != {"R", "G", "B"}:
        raise ValueError(f"Invalid Bayer pattern: '{pattern}'. "
                         "Expected one of RGGB, BGGR, GRBG, GBRG.")

    H, W = bayer.shape
    R = np.zeros((H, W), dtype=np.float32)
    G = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)

    # Locate each channel in the 2×2 super-pixel
    r0, r1 = divmod(pattern.index("R"), 2)
    b0, b1 = divmod(pattern.index("B"), 2)
    g_first  = pattern.index("G")
    g_second = pattern.index("G", g_first + 1)
    g0, g1   = divmod(g_first,  2)
    g2, g3   = divmod(g_second, 2)

    # Place known mosaic values
    R[r0::2, r1::2] = bayer[r0::2, r1::2]
    G[g0::2, g1::2] = bayer[g0::2, g1::2]
    G[g2::2, g3::2] = bayer[g2::2, g3::2]
    B[b0::2, b1::2] = bayer[b0::2, b1::2]

    # Interpolate missing pixels
    def _fill(channel, kernel):
        interp = convolve(channel, kernel, mode="mirror")
        mask   = (channel == 0).astype(np.float32)
        return channel * (1 - mask) + interp * mask

    R_full = _fill(R, _KERNEL_RB)
    B_full = _fill(B, _KERNEL_RB)
    G_full = _fill(G, _KERNEL_G)

    return np.clip(np.stack([R_full, G_full, B_full], axis=2), 0.0, 1.0)
