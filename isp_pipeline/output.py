"""
Stages 9–11 — sRGB Gamma + Resize + JPEG Writer
=================================================

Stage 9   Apply IEC 61966-2-1 sRGB transfer function (gamma encoding).

Stage 10  Bilinear image resize and digital zoom, implemented from scratch
          using NumPy advanced indexing — no external imaging library.

Stage 11  Baseline sequential DCT JPEG encoder written from scratch:
            • 8×8 DCT-II (separable matrix form)
            • Standard JPEG quantisation tables (Annex K)
            • Zigzag scan
            • Huffman coding with JPEG spec tables
            • JFIF container assembly using `struct`
"""

import struct
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Stage 9 — sRGB Gamma
# ══════════════════════════════════════════════════════════════════════════════

def apply_srgb_gamma(img: np.ndarray) -> np.ndarray:
    """
    Apply the IEC 61966-2-1 sRGB piecewise gamma transfer function.

    Parameters
    ----------
    img : (H, W, 3) float32 linear RGB in [0, 1]

    Returns
    -------
    (H, W, 3) float32 gamma-encoded sRGB in [0, 1]
    """
    print("[9] Output colour space conversion -> sRGB (gamma) ...")
    lin  = np.clip(img, 0.0, None)
    srgb = np.where(
        lin <= 0.0031308,
        12.92 * lin,
        1.055 * np.power(np.clip(lin, 0.0031308, None), 1.0 / 2.4) - 0.055,
    )
    return srgb.clip(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 10 — Bilinear Resize + Digital Zoom
# ══════════════════════════════════════════════════════════════════════════════

def _bilinear_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear interpolation resize — pure NumPy."""
    in_h, in_w = img.shape[:2]
    r  = np.arange(out_h, dtype=np.float32) * ((in_h - 1) / max(out_h - 1, 1))
    c  = np.arange(out_w, dtype=np.float32) * ((in_w - 1) / max(out_w - 1, 1))
    r0 = np.floor(r).astype(int).clip(0, in_h - 2)
    c0 = np.floor(c).astype(int).clip(0, in_w - 2)
    dr = (r - r0)[:, np.newaxis]   # (out_h, 1)
    dc = (c - c0)[np.newaxis, :]   # (1,     out_w)

    out = np.empty((out_h, out_w, img.shape[2]), dtype=np.float32)
    for ch in range(img.shape[2]):
        I = img[..., ch]
        out[..., ch] = (
            I[np.ix_(r0,     c0    )] * (1 - dr) * (1 - dc)
          + I[np.ix_(r0,     c0 + 1)] * (1 - dr) * dc
          + I[np.ix_(r0 + 1, c0    )] * dr        * (1 - dc)
          + I[np.ix_(r0 + 1, c0 + 1)] * dr        * dc
        )
    return out


def resize_image(
    img: np.ndarray,
    output_width:  int   | None = None,
    output_height: int   | None = None,
    zoom:          float = 1.0,
) -> np.ndarray:
    """
    Optionally crop-zoom and/or resize the image.

    Parameters
    ----------
    img           : (H, W, 3) float32
    output_width  : target width  in pixels  (None = keep current)
    output_height : target height in pixels  (None = keep current)
    zoom          : digital zoom factor ≥ 1.0  (centre-crop then upscale)

    Returns
    -------
    (out_h, out_w, 3) float32
    """
    print(f"[10] Image resizing (zoom={zoom:.2f}) ...")
    H, W = img.shape[:2]

    # Centre-crop for digital zoom
    if zoom != 1.0:
        ch, cw = int(H / zoom), int(W / zoom)
        y0, x0 = (H - ch) // 2, (W - cw) // 2
        img    = img[y0: y0 + ch, x0: x0 + cw]
        H, W   = img.shape[:2]

    ow = output_width  or W
    oh = output_height or H
    if (ow, oh) != (W, H):
        img = _bilinear_resize(img, oh, ow)

    return img


# ══════════════════════════════════════════════════════════════════════════════
# Stage 11 — JPEG Encoder
# ══════════════════════════════════════════════════════════════════════════════

# ── Quantisation tables (JPEG spec Annex K) ───────────────────────────────────

_QT_LUM = np.array([
    16,11,10,16, 24, 40, 51, 61,
    12,12,14,19, 26, 58, 60, 55,
    14,13,16,24, 40, 57, 69, 56,
    14,17,22,29, 51, 87, 80, 62,
    18,22,37,56, 68,109,103, 77,
    24,35,55,64, 81,104,113, 92,
    49,64,78,87,103,121,120,101,
    72,92,95,98,112,100,103, 99,
], dtype=np.float32).reshape(8, 8)

_QT_CHR = np.array([
    17,18,24,47,99,99,99,99,
    18,21,26,66,99,99,99,99,
    24,26,56,99,99,99,99,99,
    47,66,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
], dtype=np.float32).reshape(8, 8)

# ── Zigzag scan order ─────────────────────────────────────────────────────────

_ZIGZAG = [
     0, 1, 8,16, 9, 2, 3,10,17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63,
]

# ── Standard Huffman tables (JPEG spec Annex K) ───────────────────────────────

_DC_LUM_BITS = [0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0]
_DC_LUM_VALS = [0,1,2,3,4,5,6,7,8,9,10,11]
_DC_CHR_BITS = [0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
_DC_CHR_VALS = [0,1,2,3,4,5,6,7,8,9,10,11]

_AC_LUM_BITS = [0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125]
_AC_LUM_VALS = [
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa,
]

_AC_CHR_BITS = [0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119]
_AC_CHR_VALS = [
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa,
]


# ── DCT ───────────────────────────────────────────────────────────────────────

def _dct8(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II of an 8×8 block via separable 1-D DCT matrix multiply."""
    N = 8
    n = np.arange(N)
    k = np.arange(N)
    C = np.cos(np.pi * np.outer(k, 2 * n + 1) / (2 * N))
    C[0] /= np.sqrt(2)
    C    *= np.sqrt(2 / N)
    return C @ block @ C.T


# ── Huffman ───────────────────────────────────────────────────────────────────

def _build_huffman(bits: list, vals: list) -> dict:
    """Build symbol → (code_length, code_int) lookup from BITS/VALS arrays."""
    codes = {}
    code, vi = 0, 0
    for length, count in enumerate(bits, 1):
        for _ in range(count):
            codes[vals[vi]] = (length, code)
            vi += 1
            code += 1
        code <<= 1
    return codes


def _bit_length(val: int) -> int:
    return 0 if val == 0 else int(np.floor(np.log2(abs(val)))) + 1


class _BitWriter:
    """Accumulate bits into bytes, inserting 0x00 stuffing after 0xFF."""

    def __init__(self):
        self.buf   = bytearray()
        self._byte = 0
        self._bits = 0

    def write(self, val: int, n: int):
        for i in range(n - 1, -1, -1):
            self._byte = (self._byte << 1) | ((val >> i) & 1)
            self._bits += 1
            if self._bits == 8:
                self.buf.append(self._byte)
                if self._byte == 0xFF:
                    self.buf.append(0x00)   # byte stuffing
                self._byte = self._bits = 0

    def flush(self) -> bytes:
        if self._bits:
            self._byte <<= (8 - self._bits)
            self.buf.append(self._byte)
            if self._byte == 0xFF:
                self.buf.append(0x00)
        return bytes(self.buf)


def _encode_block(
    zz: np.ndarray,
    dc_prev: int,
    dc_huff: dict,
    ac_huff: dict,
    bw: _BitWriter,
) -> int:
    """Huffman-encode one 8×8 DCT block (zigzag order). Returns new DC value."""
    # DC coefficient (DPCM encoded)
    diff = int(zz[0]) - dc_prev
    cat  = _bit_length(abs(diff))
    ln, code = dc_huff[cat]
    bw.write(code, ln)
    if cat > 0:
        bw.write(diff if diff > 0 else diff + (1 << cat) - 1, cat)

    # AC coefficients
    i = 1
    while i < 64:
        zeros = 0
        while i < 64 and zz[i] == 0:
            zeros += 1
            i     += 1
        if i == 64:
            ln, code = ac_huff[0x00]    # EOB
            bw.write(code, ln)
            break
        while zeros > 15:               # ZRL (16 zeros)
            ln, code = ac_huff[0xF0]
            bw.write(code, ln)
            zeros -= 16
        val = int(zz[i])
        cat = _bit_length(abs(val))
        ln, code = ac_huff[(zeros << 4) | cat]
        bw.write(code, ln)
        bw.write(val if val > 0 else val + (1 << cat) - 1, cat)
        i += 1

    return int(zz[0])


# ── JFIF segment builders ─────────────────────────────────────────────────────

def _seg(marker: int, data: bytes) -> bytes:
    return struct.pack(">HH", marker, len(data) + 2) + data

def _qt_seg(table: np.ndarray, tid: int) -> bytes:
    flat = np.array([table.flat[_ZIGZAG[i]] for i in range(64)], dtype=np.uint8)
    return _seg(0xFFDB, bytes([tid]) + flat.tobytes())

def _dht_seg(bits: list, vals: list, tc: int, th: int) -> bytes:
    return _seg(0xFFC4, bytes([tc << 4 | th]) + bytes(bits) + bytes(vals))

def _sof0_seg(H: int, W: int) -> bytes:
    d = struct.pack(">BHHB", 8, H, W, 3)
    for i, qt in enumerate([0, 1, 1]):
        d += struct.pack("BBB", i + 1, 0x11, qt)
    return _seg(0xFFC0, d)

def _sos_seg() -> bytes:
    d = struct.pack("B", 3)
    for i, (dc, ac) in enumerate([(0, 0), (1, 1), (1, 1)]):
        d += struct.pack("BB", i + 1, (dc << 4) | ac)
    return _seg(0xFFDA, d + struct.pack("BBB", 0, 63, 0))


# ── Colour conversion ─────────────────────────────────────────────────────────

def _rgb_to_ycbcr(img_u8: np.ndarray) -> np.ndarray:
    R = img_u8[..., 0].astype(np.float32)
    G = img_u8[..., 1].astype(np.float32)
    B = img_u8[..., 2].astype(np.float32)
    Y  =  0.299   * R + 0.587   * G + 0.114   * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5     * B + 128
    Cr =  0.5     * R - 0.41869 * G - 0.08131 * B + 128
    return np.stack([Y, Cb, Cr], axis=-1)


# ── Public API ────────────────────────────────────────────────────────────────

def save_jpeg(img: np.ndarray, path: str, quality: int = 92) -> None:
    """
    Save an sRGB image as a baseline JPEG file.

    The encoder is written entirely from scratch:
      RGB → YCbCr → 8×8 block DCT → quantise → Huffman → JFIF bytes

    Parameters
    ----------
    img     : (H, W, 3) float32 sRGB in [0, 1]
    path    : output file path (should end in .jpg / .jpeg)
    quality : JPEG quality factor 1–100 (default 92)
    """
    print(f"[11] JPEG compression + save -> {path}  (quality={quality}) ...")

    # Scale quantisation tables according to quality factor
    q     = max(1, min(100, quality))
    scale = (5000 / q) if q < 50 else (200 - 2 * q)
    qt_y  = np.clip(np.floor((_QT_LUM * scale + 50) / 100), 1, 255)
    qt_c  = np.clip(np.floor((_QT_CHR * scale + 50) / 100), 1, 255)

    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    H, W   = img_u8.shape[:2]

    # Pad to multiple of 8
    pH = (H + 7) // 8 * 8
    pW = (W + 7) // 8 * 8
    if pH != H or pW != W:
        pad          = np.zeros((pH, pW, 3), dtype=np.uint8)
        pad[:H, :W]  = img_u8
        img_u8       = pad

    ycbcr   = _rgb_to_ycbcr(img_u8)
    dc_lum  = _build_huffman(_DC_LUM_BITS, _DC_LUM_VALS)
    dc_chr  = _build_huffman(_DC_CHR_BITS, _DC_CHR_VALS)
    ac_lum  = _build_huffman(_AC_LUM_BITS, _AC_LUM_VALS)
    ac_chr  = _build_huffman(_AC_CHR_BITS, _AC_CHR_VALS)
    bw      = _BitWriter()
    dc_prev = [0, 0, 0]

    for by in range(0, pH, 8):
        for bx in range(0, pW, 8):
            for ch in range(3):
                block = ycbcr[by: by+8, bx: bx+8, ch].astype(np.float32) - 128.0
                qt    = qt_y if ch == 0 else qt_c
                qz    = np.round(_dct8(block) / qt).astype(np.int32)
                zz    = np.array([qz.flat[_ZIGZAG[i]] for i in range(64)])
                dc_prev[ch] = _encode_block(
                    zz, dc_prev[ch],
                    dc_lum if ch == 0 else dc_chr,
                    ac_lum if ch == 0 else ac_chr,
                    bw,
                )

    scan = bw.flush()

    jfif = (b"\xFF\xE0"
            + struct.pack(">H", 16)
            + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")

    payload = (
        b"\xFF\xD8"                                       # SOI
        + jfif                                            # APP0
        + _qt_seg(qt_y, 0) + _qt_seg(qt_c, 1)            # DQT
        + _sof0_seg(H, W)                                 # SOF0
        + _dht_seg(_DC_LUM_BITS, _DC_LUM_VALS, 0, 0)     # DHT
        + _dht_seg(_AC_LUM_BITS, _AC_LUM_VALS, 1, 0)
        + _dht_seg(_DC_CHR_BITS, _DC_CHR_VALS, 0, 1)
        + _dht_seg(_AC_CHR_BITS, _AC_CHR_VALS, 1, 1)
        + _sos_seg()                                      # SOS header
        + scan                                            # entropy-coded data
        + b"\xFF\xD9"                                     # EOI
    )

    with open(path, "wb") as f:
        f.write(payload)

    print(f"     + Saved: {path}")
