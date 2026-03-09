"""
Stages 7 & 8 — Noise Reduction + Sharpening
=============================================

Stage 7  Gaussian noise reduction via `scipy.ndimage.gaussian_filter`.
         Applied per-channel in linear RGB space.

Stage 8  Unsharp mask sharpening:
           output = input + amount × (input − blurred)
         Uses the same Gaussian as the denoising stage.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def noise_reduction(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Reduce noise with a per-channel Gaussian blur.

    Parameters
    ----------
    img   : (H, W, 3) float32 in [0, 1]
    sigma : standard deviation of the Gaussian kernel
            (larger → more smoothing, more detail loss)

    Returns
    -------
    (H, W, 3) float32 in [0, 1]
    """
    print(f"[7] Noise reduction (Gaussian sigma={sigma}) ...")
    denoised = np.stack(
        [gaussian_filter(img[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )
    return np.clip(denoised, 0.0, 1.0)


def sharpening(img: np.ndarray, amount: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    """
    Sharpen using an unsharp mask.

    Parameters
    ----------
    img    : (H, W, 3) float32 in [0, 1]
    amount : sharpening strength  (0 = none, 1 = moderate, 2 = strong)
    sigma  : radius of the blur used to extract the detail layer

    Returns
    -------
    (H, W, 3) float32 in [0, 1]
    """
    print(f"[8] Sharpening (unsharp mask, amount={amount}) ...")
    blurred = np.stack(
        [gaussian_filter(img[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )
    # detail = high-frequency residual
    return np.clip(img + amount * (img - blurred), 0.0, 1.0)
