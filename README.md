# Single-Frame Camera ISP Pipeline

A from-scratch implementation of a single-frame camera Image Signal Processing (ISP) pipeline in Python, using only **NumPy** and **SciPy**.

```
RAW TIFF → Demosaic → White Balance → Color Transform → Color Manipulation
         → Tone Mapping → Noise Reduction → Sharpening → sRGB → Resize → JPEG
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `tiff_reader.py`     | Parse uncompressed 16-bit TIFF (no external lib) |
| 2 | `demosaic.py`        | Bilinear Bayer demosaicing (RGGB/BGGR/GRBG/GBRG) |
| 3 | `white_balance.py`   | Grey-world or manual RGB gain white balance |
| 4 | `color_transform.py` | Camera RGB → CIE XYZ D65 matrix transform |
| 5 | `color_transform.py` | Saturation / hue manipulation in HSV (from scratch) |
| 6 | `tone_mapping.py`    | Reinhard or filmic tone mapping curve |
| 7 | `enhance.py`         | Gaussian noise reduction (SciPy) |
| 8 | `enhance.py`         | Unsharp mask sharpening (SciPy) |
| 9 | `output.py`          | IEC 61966-2-1 sRGB gamma encoding |
| 10| `output.py`          | Bilinear resize + digital zoom (from scratch) |
| 11| `output.py`          | Baseline DCT JPEG writer (from scratch) |

## Installation

```bash
git clone https://github.com/TungChihYuan/BetterPhoto.git
cd BetterPhoto
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Basic usage
python main.py input.tif output.jpg

# Full options
python main.py input.tif output.jpg \
    --pattern  RGGB      \   # Bayer pattern: RGGB | BGGR | GRBG | GBRG
    --black    0.0        \   # Black level (normalised 0–1)
    --white    1.0        \   # White level (normalised 0–1)
    --saturation 1.2      \   # Colour saturation multiplier
    --hue      0.0        \   # Hue shift in degrees
    --tone     reinhard   \   # Tone mapping: reinhard | filmic
    --denoise  1.0        \   # Gaussian sigma for denoising
    --sharpen  1.5        \   # Unsharp mask strength
    --zoom     1.0        \   # Digital zoom factor (≥1.0)
    --width    1920       \   # Output width  (optional)
    --height   1080       \   # Output height (optional)
    --quality  92             # JPEG quality 1–100
```

### Python API

```python
from isp_pipeline.pipeline import run_pipeline

run_pipeline(
    input_path="photo.tif",
    output_path="photo.jpg",
    bayer_pattern="RGGB",
    saturation=1.3,
    tone_method="filmic",
    jpeg_quality=95,
)
```

## Dependencies

```
numpy
scipy
```

No rawpy, OpenCV, or Pillow required.

## Project Structure

```
isp_pipeline/
├── README.md
├── requirements.txt
├── main.py                      # CLI entry point
├── isp_pipeline/
│   ├── __init__.py
│   ├── pipeline.py              # Orchestrates all stages
│   ├── tiff_reader.py           # Stage 1:  TIFF parser
│   ├── demosaic.py              # Stage 2:  Bayer demosaicing
│   ├── white_balance.py         # Stage 3:  White balance
│   ├── color_transform.py       # Stages 4–5: XYZ transform + HSV manipulation
│   ├── tone_mapping.py          # Stage 6:  Tone mapping
│   ├── enhance.py               # Stages 7–8: Denoise + sharpen
│   └── output.py                # Stages 9–11: Gamma + resize + JPEG
└── tests/
    └── test_pipeline.py
```
## Sample Usage

```

python main.py input_photo/IMG_3189.png output.jpg --save-stages 
