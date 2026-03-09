"""
png_reader.py — From-scratch PNG reader for Bayer RAW images
=============================================================
Supports:
  • 8-bit  grayscale  (mode L)       → single-channel Bayer
  • 16-bit grayscale  (bit depth 16) → single-channel Bayer (most RAW dumps)
  • 8-bit  RGB        (mode RGB)     → converted to grayscale luminance
  • 16-bit RGB                       → converted to grayscale luminance

Uses only Python stdlib: zlib + struct.  No Pillow, no OpenCV.
"""

import zlib
import struct
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# PNG constants
# ──────────────────────────────────────────────────────────────────────────────

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# colour_type values
_CT_GRAY  = 0
_CT_RGB   = 2
_CT_GRAYA = 4
_CT_RGBA  = 6


# ──────────────────────────────────────────────────────────────────────────────
# PNG filter reconstruction (applied per scanline)
# ──────────────────────────────────────────────────────────────────────────────

def _paeth(a, b, c):
    p  = int(a) + int(b) - int(c)
    pa = abs(p - int(a))
    pb = abs(p - int(b))
    pc = abs(p - int(c))
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    return c


def _unfilter(raw_rows: list[bytes], width: int, bpp: int) -> list[bytearray]:
    """
    Undo PNG row filters.

    Parameters
    ----------
    raw_rows : list of bytes, each row prefixed with its filter byte
    width    : image width in pixels
    bpp      : bytes per pixel  (e.g. 1 for 8-bit gray, 2 for 16-bit gray,
                                      3 for 8-bit RGB, 6 for 16-bit RGB)
    """
    stride = width * bpp
    result = []
    prev   = bytearray(stride)          # previous reconstructed row (zeros for first)

    for row_raw in raw_rows:
        ftype = row_raw[0]
        fdata = bytearray(row_raw[1:])  # filtered pixel data (length = stride)
        recon = bytearray(stride)

        if ftype == 0:                  # None
            recon[:] = fdata

        elif ftype == 1:                # Sub
            for i in range(stride):
                a = recon[i - bpp] if i >= bpp else 0
                recon[i] = (fdata[i] + a) & 0xFF

        elif ftype == 2:                # Up
            for i in range(stride):
                recon[i] = (fdata[i] + prev[i]) & 0xFF

        elif ftype == 3:                # Average
            for i in range(stride):
                a = recon[i - bpp] if i >= bpp else 0
                b = prev[i]
                recon[i] = (fdata[i] + (a + b) // 2) & 0xFF

        elif ftype == 4:                # Paeth
            for i in range(stride):
                a = recon[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp]  if i >= bpp else 0
                recon[i] = (fdata[i] + _paeth(a, b, c)) & 0xFF

        else:
            raise ValueError(f"Unknown PNG filter type: {ftype}")

        result.append(recon)
        prev = recon

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def read_png_bayer(path: str) -> tuple[np.ndarray, dict]:
    """
    Load a PNG file and return a single-channel Bayer float32 array.

    Handles:
      • 8-bit  / 16-bit grayscale  →  used as-is
      • 8-bit  / 16-bit RGB        →  converted to luminance
                                      Y = 0.2126 R + 0.7152 G + 0.0722 B

    Parameters
    ----------
    path : path to the PNG file (e.g. 'raw_data_RGGB.png')

    Returns
    -------
    bayer : (H, W) float32  in [0, 1]
    info  : dict with keys  width, height, bit_depth, colour_type
    """
    with open(path, "rb") as f:
        data = f.read()

    # ── Validate signature ────────────────────────────────────────────────────
    if data[:8] != _PNG_SIG:
        raise ValueError(f"Not a PNG file: {path}")

    # ── Parse chunks ──────────────────────────────────────────────────────────
    idat_chunks = []
    pos = 8
    ihdr = None

    while pos < len(data):
        length = struct.unpack_from(">I", data, pos)[0];  pos += 4
        chunk_type = data[pos: pos + 4].decode("ascii");  pos += 4
        chunk_data = data[pos: pos + length];             pos += length
        _crc       = data[pos: pos + 4];                  pos += 4   # skip CRC

        if chunk_type == "IHDR":
            ihdr = chunk_data
        elif chunk_type == "IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == "IEND":
            break

    if ihdr is None:
        raise ValueError("PNG missing IHDR chunk")

    # ── Decode IHDR ───────────────────────────────────────────────────────────
    width, height = struct.unpack_from(">II", ihdr, 0)
    bit_depth     = ihdr[8]
    colour_type   = ihdr[9]
    # interlace method at ihdr[12] — we only support non-interlaced
    if ihdr[12] != 0:
        raise ValueError("Interlaced PNGs are not supported.")

    info = dict(width=width, height=height,
                bit_depth=bit_depth, colour_type=colour_type)

    # ── Decompress IDAT ───────────────────────────────────────────────────────
    compressed = b"".join(idat_chunks)
    raw        = zlib.decompress(compressed)

    # ── Determine bytes-per-pixel ─────────────────────────────────────────────
    channels = {_CT_GRAY: 1, _CT_RGB: 3,
                _CT_GRAYA: 2, _CT_RGBA: 4}.get(colour_type)
    if channels is None:
        raise ValueError(f"Unsupported PNG colour type: {colour_type}")

    bytes_per_sample = bit_depth // 8
    bpp              = channels * bytes_per_sample
    stride           = width * bpp

    # ── Split into rows (each row is 1 + stride bytes in raw stream) ──────────
    row_size = 1 + stride
    raw_rows = [raw[i * row_size: (i + 1) * row_size] for i in range(height)]

    # ── Undo PNG filters ──────────────────────────────────────────────────────
    rows = _unfilter(raw_rows, width, bpp)

    # ── Convert to numpy array ────────────────────────────────────────────────
    flat = bytearray()
    for r in rows:
        flat.extend(r)
    buf = bytes(flat)

    if bit_depth == 16:
        # Big-endian 16-bit samples
        n_samples = height * width * channels
        fmt       = f">{n_samples}H"
        samples   = np.array(struct.unpack(fmt, buf), dtype=np.float32)
        samples  /= 65535.0
    elif bit_depth == 8:
        samples = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    pixels = samples.reshape(height, width, channels)

    # ── Convert to single-channel Bayer ──────────────────────────────────────
    if colour_type in (_CT_GRAY, _CT_GRAYA):
        # Grayscale (+ optional alpha): just take the first channel
        bayer = pixels[..., 0]
    else:
        # RGB or RGBA: convert to luminance
        R, G, B = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        bayer   = 0.2126 * R + 0.7152 * G + 0.0722 * B

    return bayer.astype(np.float32), info
