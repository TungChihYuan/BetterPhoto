"""
Stages 4 & 5 — Colour Space Transform + Colour Manipulation
=============================================================

Stage 4  camera/sRGB linear RGB  →  CIE XYZ D65
         Uses the standard IEC 61966-2-1 sRGB primary matrix.
         For a more accurate result, replace _RGB_TO_XYZ with your
         camera's ColorMatrix2 tag from its EXIF data.

Stage 5  Saturation and hue adjustment in HSV space.
         HSV ↔ RGB conversion is implemented from scratch (no OpenCV).
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Colour matrices  (sRGB primaries, D65 white point)
# ──────────────────────────────────────────────────────────────────────────────

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)

_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — RGB ↔ XYZ
# ──────────────────────────────────────────────────────────────────────────────

def rgb_to_xyz(img: np.ndarray) -> np.ndarray:
    """
    Convert linear sRGB → CIE XYZ D65.

    Parameters
    ----------
    img : (H, W, 3) float32 linear RGB in [0, 1]

    Returns
    -------
    (H, W, 3) float32 XYZ  (values may exceed 1.0)
    """
    print("[4] Colour space transform -> CIE XYZ D65 ...")
    H, W, _ = img.shape
    return (img.reshape(-1, 3) @ _RGB_TO_XYZ.T).reshape(H, W, 3).clip(0)


def xyz_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert CIE XYZ D65 → linear sRGB (clipped to [0, 1])."""
    H, W, _ = img.shape
    return (img.reshape(-1, 3) @ _XYZ_TO_RGB.T).reshape(H, W, 3).clip(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — HSV helpers (from scratch)
# ──────────────────────────────────────────────────────────────────────────────

def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Vectorised RGB → HSV.  H in [0, 360), S and V in [0, 1]."""
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Cmax  = np.maximum(np.maximum(R, G), B)
    Cmin  = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin + 1e-8

    H = np.zeros_like(R)
    m = Cmax == R;  H[m] = (60 * ((G[m] - B[m]) / delta[m])) % 360
    m = Cmax == G;  H[m] = (60 * ((B[m] - R[m]) / delta[m]) + 120) % 360
    m = Cmax == B;  H[m] = (60 * ((R[m] - G[m]) / delta[m]) + 240) % 360

    S = np.where(Cmax > 0, delta / Cmax, 0.0)
    return np.stack([H, S, Cmax], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV → RGB.  H in [0, 360), S/V in [0, 1]."""
    H = hsv[..., 0] % 360
    S = hsv[..., 1]
    V = hsv[..., 2]

    i = (H / 60).astype(int) % 6
    f = H / 60 - np.floor(H / 60)
    p = V * (1 - S)
    q = V * (1 - f * S)
    t = V * (1 - (1 - f) * S)

    R = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [V, q, p, p, t, V])
    G = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [t, V, V, q, p, p])
    B = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [p, p, t, V, V, q])

    return np.clip(np.stack([R, G, B], axis=-1), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — Public API
# ──────────────────────────────────────────────────────────────────────────────

def color_manipulation(
    img_xyz: np.ndarray,
    saturation: float = 1.2,
    hue_shift: float  = 0.0,
) -> np.ndarray:
    """
    Adjust saturation and hue of an XYZ image.

    Parameters
    ----------
    img_xyz    : (H, W, 3) float32 CIE XYZ
    saturation : multiplier for the S channel in HSV (1.0 = no change)
    hue_shift  : degrees to rotate the H channel (0.0 = no change)

    Returns
    -------
    (H, W, 3) float32 linear sRGB in [0, 1]
    """
    print("[5] Colour manipulation (saturation / hue) ...")
    rgb = xyz_to_rgb(img_xyz)
    hsv = _rgb_to_hsv(rgb)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
    return _hsv_to_rgb(hsv)
