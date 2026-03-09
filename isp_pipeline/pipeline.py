"""
pipeline.py — Orchestrates all ISP stages end-to-end.
Supports both TIFF and PNG input formats.

If save_stages=True, each intermediate stage is saved as a JPEG into
a subfolder next to the output file, e.g.:
    output_stages/
        stage_01_raw_preprocess.jpg
        stage_02_bayer_demosaic.jpg
        ...
        stage_11_final.jpg
"""

import os
import numpy as np

from .tiff_reader      import read_tiff, raw_preprocess
from .png_reader       import read_png_bayer
from .demosaic         import bayer_demosaic
from .white_balance    import white_balance
from .color_transform  import rgb_to_xyz, color_manipulation
from .tone_mapping     import tone_mapping
from .enhance          import noise_reduction, sharpening
from .output           import apply_srgb_gamma, resize_image, save_jpeg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_bayer(input_path: str, black_level: float, white_level: float) -> np.ndarray:
    """Auto-detect input format (TIFF or PNG) and return a normalised Bayer array."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".tif", ".tiff"):
        print(f"[0] Detected TIFF input: {input_path}")
        bayer, _ = read_tiff(input_path)
    elif ext == ".png":
        print(f"[0] Detected PNG input: {input_path}")
        bayer, info = read_png_bayer(input_path)
        print(f"     size={info['width']}x{info['height']}, "
              f"bit_depth={info['bit_depth']}, colour_type={info['colour_type']}")
    else:
        raise ValueError(
            f"Unsupported input format: '{ext}'\n"
            "Supported formats: .tif, .tiff, .png"
        )
    return raw_preprocess(bayer, black_level, white_level)


def _to_display(img: np.ndarray) -> np.ndarray:
    """
    Convert any intermediate image to a display-ready float32 RGB in [0,1].

    - 2-D (H, W)   → grayscale repeated to (H, W, 3)
    - 3-D (H, W, 3)→ passed through; values clipped & normalised
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)   # stretch to [0,1] so dark stages are visible
    return np.clip(img, 0.0, 1.0)


def _save_stage(img: np.ndarray, stage_dir: str, stage_num: int,
                stage_name: str, quality: int = 92) -> None:
    """Normalise img to display range and save as JPEG."""
    os.makedirs(stage_dir, exist_ok=True)
    fname = os.path.join(stage_dir, f"stage_{stage_num:02d}_{stage_name}.jpg")
    display = _to_display(img)
    save_jpeg(display, fname, quality=quality)
    # suppress the verbose print from save_jpeg — reprint ourselves
    print(f"     → saved stage image: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_path:     str,
    output_path:    str,
    bayer_pattern:  str   = "RGGB",
    black_level:    float = 0.0,
    white_level:    float = 1.0,
    saturation:     float = 1.2,
    hue_shift:      float = 0.0,
    tone_method:    str   = "reinhard",
    denoise_sigma:  float = 1.0,
    sharpen_amount: float = 1.5,
    zoom:           float = 1.0,
    output_width:   int   | None = None,
    output_height:  int   | None = None,
    jpeg_quality:   int   = 92,
    save_stages:    bool  = False,
) -> None:
    """
    Run the full single-frame ISP pipeline.

    Parameters
    ----------
    input_path     : path to input file (.tif / .tiff / .png)
    output_path    : path for the final output JPEG
    bayer_pattern  : 'RGGB' | 'BGGR' | 'GRBG' | 'GBRG'
    black_level    : normalised black point (default 0.0)
    white_level    : normalised white point (default 1.0)
    saturation     : colour saturation multiplier (1.0 = neutral)
    hue_shift      : hue rotation in degrees (0.0 = neutral)
    tone_method    : 'reinhard' | 'filmic'
    denoise_sigma  : Gaussian sigma for noise reduction
    sharpen_amount : unsharp mask strength
    zoom           : digital zoom factor >= 1.0
    output_width   : resize output width  (None = no resize)
    output_height  : resize output height (None = no resize)
    jpeg_quality   : JPEG quality 1-100
    save_stages    : if True, save every intermediate stage as a JPEG
                     in a subfolder  <output_basename>_stages/
    """
    print("\n==========================================")
    print("  Single-Frame Camera ISP Pipeline")
    print("==========================================\n")

    # Prepare stages output directory
    if save_stages:
        base = os.path.splitext(output_path)[0]
        stage_dir = base + "_stages"
        os.makedirs(stage_dir, exist_ok=True)
        print(f"Stage images will be saved to: {stage_dir}/\n")

    def maybe_save(img, num, name):
        if save_stages:
            _save_stage(img, stage_dir, num, name, quality=jpeg_quality)

    # ── Stage 1: Raw pre-processing ───────────────────────────────────────────
    bayer = _load_bayer(input_path, black_level, white_level)
    maybe_save(bayer, 1, "raw_preprocess")

    # ── Stage 2: Bayer demosaicing ────────────────────────────────────────────
    img = bayer_demosaic(bayer, pattern=bayer_pattern)
    maybe_save(img, 2, "bayer_demosaic")

    # ── Stage 3: White balance ────────────────────────────────────────────────
    img = white_balance(img)
    maybe_save(img, 3, "white_balance")

    # ── Stage 4: Colour space transform → XYZ ────────────────────────────────
    img_xyz = rgb_to_xyz(img)
    maybe_save(img_xyz, 4, "color_space_xyz")

    # ── Stage 5: Colour manipulation ──────────────────────────────────────────
    img = color_manipulation(img_xyz, saturation, hue_shift)
    maybe_save(img, 5, "color_manipulation")

    # ── Stage 6: Tone mapping ─────────────────────────────────────────────────
    img = tone_mapping(img, method=tone_method)
    maybe_save(img, 6, "tone_mapping")

    # ── Stage 7: Noise reduction ──────────────────────────────────────────────
    img = noise_reduction(img, sigma=denoise_sigma)
    maybe_save(img, 7, "noise_reduction")

    # ── Stage 8: Sharpening ───────────────────────────────────────────────────
    img = sharpening(img, amount=sharpen_amount)
    maybe_save(img, 8, "sharpening")

    # ── Stage 9: sRGB gamma ───────────────────────────────────────────────────
    img = apply_srgb_gamma(img)
    maybe_save(img, 9, "srgb_gamma")

    # ── Stage 10: Resize + digital zoom ──────────────────────────────────────
    img = resize_image(img, output_width, output_height, zoom)
    maybe_save(img, 10, "resize")

    # ── Stage 11: Final JPEG output ───────────────────────────────────────────
    print("[11] JPEG compression + save ...")
    save_jpeg(img, output_path, quality=jpeg_quality)
    if save_stages:
        import shutil
        final_stage = os.path.join(stage_dir, "stage_11_final.jpg")
        shutil.copy(output_path, final_stage)
        print(f"     → saved stage image: {final_stage}")

    print(f"\n✅ Pipeline complete!")
    if save_stages:
        print(f"   Stage images saved in: {stage_dir}/")
    print()
