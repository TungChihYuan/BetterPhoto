"""
tests/test_pipeline.py
======================
Unit tests for each ISP pipeline stage.
Run with:  python -m pytest tests/
"""

import struct
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isp_pipeline.tiff_reader     import raw_preprocess
from isp_pipeline.demosaic        import bayer_demosaic
from isp_pipeline.white_balance   import white_balance
from isp_pipeline.color_transform import rgb_to_xyz, xyz_to_rgb, color_manipulation
from isp_pipeline.tone_mapping    import tone_mapping
from isp_pipeline.enhance         import noise_reduction, sharpening
from isp_pipeline.output          import apply_srgb_gamma, resize_image, save_jpeg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rand_rgb(H=64, W=64):
    rng = np.random.default_rng(42)
    return rng.random((H, W, 3), dtype=np.float64).astype(np.float32)

def _synthetic_bayer(H=64, W=64, pattern="RGGB"):
    """Create a simple synthetic Bayer image from a known RGB image."""
    rng = np.random.default_rng(0)
    rgb = rng.random((H, W, 3), dtype=np.float32)
    bayer = np.zeros((H, W), dtype=np.float32)
    r0, r1 = divmod(pattern.index("R"), 2)
    b0, b1 = divmod(pattern.index("B"), 2)
    g0, g1 = divmod(pattern.index("G"), 2)
    g2, g3 = divmod(pattern.index("G", pattern.index("G") + 1), 2)
    bayer[r0::2, r1::2] = rgb[r0::2, r1::2, 0]
    bayer[g0::2, g1::2] = rgb[g0::2, g1::2, 1]
    bayer[g2::2, g3::2] = rgb[g2::2, g3::2, 1]
    bayer[b0::2, b1::2] = rgb[b0::2, b1::2, 2]
    return bayer


# ── Stage 1 ───────────────────────────────────────────────────────────────────

class TestRawPreprocess:
    def test_output_range(self):
        bayer = np.random.rand(64, 64).astype(np.float32)
        out   = raw_preprocess(bayer)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_black_white_levels(self):
        bayer = np.full((8, 8), 0.5, dtype=np.float32)
        out   = raw_preprocess(bayer, black_level=0.25, white_level=0.75)
        assert np.allclose(out, 0.5, atol=1e-5)

    def test_clipping(self):
        bayer = np.array([[-1.0, 2.0]], dtype=np.float32)
        out   = raw_preprocess(bayer)
        assert out.min() >= 0.0 and out.max() <= 1.0


# ── Stage 2 ───────────────────────────────────────────────────────────────────

class TestBayerDemosaic:
    @pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
    def test_output_shape(self, pattern):
        bayer = _synthetic_bayer(pattern=pattern)
        out   = bayer_demosaic(bayer, pattern=pattern)
        assert out.shape == (64, 64, 3)

    def test_output_range(self):
        bayer = _synthetic_bayer()
        out   = bayer_demosaic(bayer)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_invalid_pattern(self):
        bayer = _synthetic_bayer()
        with pytest.raises(ValueError):
            bayer_demosaic(bayer, pattern="XXXX")


# ── Stage 3 ───────────────────────────────────────────────────────────────────

class TestWhiteBalance:
    def test_grey_world_neutral(self):
        # A perfectly neutral image should remain unchanged
        img = np.full((8, 8, 3), 0.5, dtype=np.float32)
        out = white_balance(img)
        assert np.allclose(out, 0.5, atol=1e-4)

    def test_manual_gains(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out = white_balance(img, r_gain=2.0, g_gain=1.0, b_gain=0.5)
        assert np.allclose(out[..., 0], 1.0,  atol=1e-5)
        assert np.allclose(out[..., 1], 0.5,  atol=1e-5)
        assert np.allclose(out[..., 2], 0.25, atol=1e-5)

    def test_output_clipped(self):
        img = np.ones((4, 4, 3), dtype=np.float32)
        out = white_balance(img, r_gain=5.0, g_gain=1.0, b_gain=1.0)
        assert out.max() <= 1.0


# ── Stage 4 ───────────────────────────────────────────────────────────────────

class TestColorTransform:
    def test_roundtrip(self):
        img = _rand_rgb()
        xyz = rgb_to_xyz(img)
        out = xyz_to_rgb(xyz)
        assert np.allclose(img, out, atol=1e-4)

    def test_xyz_shape(self):
        img = _rand_rgb()
        xyz = rgb_to_xyz(img)
        assert xyz.shape == img.shape


# ── Stage 5 ───────────────────────────────────────────────────────────────────

class TestColorManipulation:
    def test_output_range(self):
        img = _rand_rgb()
        xyz = rgb_to_xyz(img)
        out = color_manipulation(xyz, saturation=1.5, hue_shift=30.0)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_neutral_saturation(self):
        img = _rand_rgb()
        xyz = rgb_to_xyz(img)
        out = color_manipulation(xyz, saturation=1.0, hue_shift=0.0)
        ref = xyz_to_rgb(xyz)
        assert np.allclose(out, ref, atol=1e-4)


# ── Stage 6 ───────────────────────────────────────────────────────────────────

class TestToneMapping:
    def test_reinhard_output_range(self):
        img = _rand_rgb() * 5   # HDR-like
        out = tone_mapping(img, "reinhard")
        assert out.min() >= 0.0 and out.max() < 1.0

    def test_filmic_output_range(self):
        img = _rand_rgb() * 5
        out = tone_mapping(img, "filmic")
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            tone_mapping(_rand_rgb(), "unknown")


# ── Stage 7 & 8 ───────────────────────────────────────────────────────────────

class TestEnhance:
    def test_denoise_shape(self):
        img = _rand_rgb()
        out = noise_reduction(img, sigma=1.0)
        assert out.shape == img.shape

    def test_denoise_range(self):
        img = _rand_rgb()
        out = noise_reduction(img)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_sharpen_shape(self):
        img = _rand_rgb()
        out = sharpening(img, amount=1.0)
        assert out.shape == img.shape

    def test_sharpen_range(self):
        img = _rand_rgb()
        out = sharpening(img)
        assert out.min() >= 0.0 and out.max() <= 1.0


# ── Stage 9 ───────────────────────────────────────────────────────────────────

class TestSrgbGamma:
    def test_zero_maps_to_zero(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        out = apply_srgb_gamma(img)
        assert np.allclose(out, 0.0)

    def test_one_maps_to_one(self):
        img = np.ones((4, 4, 3), dtype=np.float32)
        out = apply_srgb_gamma(img)
        assert np.allclose(out, 1.0, atol=1e-5)

    def test_output_range(self):
        img = _rand_rgb()
        out = apply_srgb_gamma(img)
        assert out.min() >= 0.0 and out.max() <= 1.0


# ── Stage 10 ──────────────────────────────────────────────────────────────────

class TestResize:
    def test_resize_shape(self):
        img = _rand_rgb(64, 64)
        out = resize_image(img, output_width=32, output_height=32)
        assert out.shape == (32, 32, 3)

    def test_no_op(self):
        img = _rand_rgb(64, 64)
        out = resize_image(img)
        assert out.shape == img.shape

    def test_zoom_shape(self):
        # zoom=2.0 crops to half size; without explicit output dims, stays cropped
        img = _rand_rgb(64, 64)
        out = resize_image(img, zoom=2.0)
        assert out.shape == (32, 32, 3)

    def test_zoom_with_output_size(self):
        # zoom + explicit output size should upscale back to requested dims
        img = _rand_rgb(64, 64)
        out = resize_image(img, output_width=64, output_height=64, zoom=2.0)
        assert out.shape == (64, 64, 3)


# ── Stage 11 ──────────────────────────────────────────────────────────────────

class TestSaveJpeg:
    def test_file_created(self, tmp_path):
        img  = _rand_rgb(32, 32)
        path = str(tmp_path / "out.jpg")
        save_jpeg(img, path, quality=80)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100

    def test_valid_jpeg_markers(self, tmp_path):
        img  = _rand_rgb(16, 16)
        path = str(tmp_path / "out.jpg")
        save_jpeg(img, path)
        with open(path, "rb") as f:
            data = f.read()
        assert data[:2]  == b"\xFF\xD8", "Missing SOI marker"
        assert data[-2:] == b"\xFF\xD9", "Missing EOI marker"
