"""
Stage 1 — Raw Pre-processing & TIFF Reader
==========================================
Parses an uncompressed single-channel TIFF (8-bit or 16-bit),
then subtracts the black level and normalises to [0, 1].

No external imaging libraries are used; only Python `struct` + NumPy.
"""

import struct
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal TIFF helpers
# ──────────────────────────────────────────────────────────────────────────────

def _u16(data: bytes, off: int, little: bool) -> int:
    return struct.unpack_from("<H" if little else ">H", data, off)[0]

def _u32(data: bytes, off: int, little: bool) -> int:
    return struct.unpack_from("<I" if little else ">I", data, off)[0]

def _parse_ifd(data: bytes, ifd_off: int, little: bool) -> dict:
    """Read all IFD tag entries and return a {tag_id: value} dict."""
    tags = {}
    n = _u16(data, ifd_off, little)
    for i in range(n):
        base     = ifd_off + 2 + i * 12
        tag      = _u16(data, base,     little)
        typ      = _u16(data, base + 2, little)
        cnt      = _u32(data, base + 4, little)
        val_off  = base + 8
        type_sz  = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}.get(typ, 1)

        if cnt * type_sz <= 4:
            raw = data[val_off: val_off + cnt * type_sz]
        else:
            ptr = _u32(data, val_off, little)
            raw = data[ptr: ptr + cnt * type_sz]

        if typ == 3:   # SHORT
            fmt = ("<" if little else ">") + "H" * cnt
            vals = list(struct.unpack(fmt, raw))
            tags[tag] = vals[0] if cnt == 1 else vals
        elif typ == 4:  # LONG
            fmt = ("<" if little else ">") + "I" * cnt
            vals = list(struct.unpack(fmt, raw))
            tags[tag] = vals[0] if cnt == 1 else vals
        else:
            tags[tag] = raw
    return tags


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def read_tiff(path: str) -> tuple[np.ndarray, dict]:
    """
    Load an uncompressed single-channel TIFF file.

    Parameters
    ----------
    path : str
        Path to a 16-bit (or 8-bit) uncompressed Bayer TIFF.

    Returns
    -------
    pixels : np.ndarray, shape (H, W), dtype float32, range [0, 1]
    tags   : dict  — raw IFD tag dictionary
    """
    with open(path, "rb") as f:
        data = f.read()

    bom = data[:2]
    if   bom == b"II": little = True
    elif bom == b"MM": little = False
    else: raise ValueError(f"Not a TIFF file: {path}")

    assert _u16(data, 2, little) == 42, "Bad TIFF magic number"

    ifd_off = _u32(data, 4, little)
    tags    = _parse_ifd(data, ifd_off, little)

    width       = tags[256]
    height      = tags[257]
    bps         = tags.get(258, 8)
    compression = tags.get(259, 1)

    if compression != 1:
        raise ValueError(
            f"Only uncompressed TIFF is supported (compression tag = {compression}).\n"
            "Convert your RAW to an uncompressed TIFF first, e.g. with dcraw:\n"
            "  dcraw -D -4 -T photo.CR2"
        )

    strip_offsets = tags[273]
    strip_bytes   = tags.get(279)
    if not isinstance(strip_offsets, list): strip_offsets = [strip_offsets]
    if strip_bytes is None:                 strip_bytes   = [len(data) - strip_offsets[0]]
    if not isinstance(strip_bytes,   list): strip_bytes   = [strip_bytes]

    raw_bytes = b"".join(data[o: o + n] for o, n in zip(strip_offsets, strip_bytes))

    if bps == 16:
        fmt    = ("<" if little else ">") + "H" * (width * height)
        pixels = np.array(struct.unpack(fmt, raw_bytes), dtype=np.float32) / 65535.0
    elif bps == 8:
        pixels = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported BitsPerSample: {bps}")

    return pixels.reshape(height, width), tags


def raw_preprocess(
    bayer: np.ndarray,
    black_level: float = 0.0,
    white_level: float = 1.0,
) -> np.ndarray:
    """
    Subtract black level and normalise to [0, 1].

    Parameters
    ----------
    bayer       : (H, W) float32 Bayer mosaic in [0, 1]
    black_level : normalised black point (default 0.0)
    white_level : normalised white point (default 1.0)
    """
    print("[1] Raw pre-processing ...")
    bayer = (bayer - black_level) / (white_level - black_level + 1e-8)
    return np.clip(bayer, 0.0, 1.0)
