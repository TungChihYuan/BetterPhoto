"""
main.py — Command-line entry point for the ISP pipeline.

Usage
-----
    python main.py input.png output.jpg [options]
    python main.py input.png output.jpg --save-stages

Run with --help for all options.
"""

import argparse
from isp_pipeline.pipeline import run_pipeline


def main():
    p = argparse.ArgumentParser(
        description="Single-Frame Camera ISP Pipeline  (NumPy + SciPy only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("input",  help="Input RAW file (.tif / .tiff / .png)")
    p.add_argument("output", help="Output JPEG path")

    # RAW decoding
    g = p.add_argument_group("RAW decoding")
    g.add_argument("--pattern", default="RGGB",
                   choices=["RGGB", "BGGR", "GRBG", "GBRG"],
                   help="Bayer CFA pattern")
    g.add_argument("--black", type=float, default=0.0,
                   help="Black level (normalised 0-1)")
    g.add_argument("--white", type=float, default=1.0,
                   help="White level (normalised 0-1)")

    # Colour grading
    g = p.add_argument_group("colour grading")
    g.add_argument("--saturation", type=float, default=1.2,
                   help="Saturation multiplier (1.0 = neutral)")
    g.add_argument("--hue", type=float, default=0.0,
                   help="Hue shift in degrees")
    g.add_argument("--tone", default="reinhard",
                   choices=["reinhard", "filmic"],
                   help="Tone-mapping operator")

    # Enhancement
    g = p.add_argument_group("enhancement")
    g.add_argument("--denoise", type=float, default=1.0,
                   help="Gaussian sigma for noise reduction (0 = off)")
    g.add_argument("--sharpen", type=float, default=1.5,
                   help="Unsharp mask strength (0 = off)")

    # Output
    g = p.add_argument_group("output")
    g.add_argument("--zoom",         type=float, default=1.0,
                   help="Digital zoom factor (>=1.0)")
    g.add_argument("--width",        type=int,   default=None,
                   help="Output width in pixels")
    g.add_argument("--height",       type=int,   default=None,
                   help="Output height in pixels")
    g.add_argument("--quality",      type=int,   default=92,
                   help="JPEG quality 1-100")
    g.add_argument("--save-stages",  action="store_true", default=False,
                   help="Save every intermediate stage as a JPEG in <output>_stages/")

    args = p.parse_args()

    run_pipeline(
        input_path     = args.input,
        output_path    = args.output,
        bayer_pattern  = args.pattern,
        black_level    = args.black,
        white_level    = args.white,
        saturation     = args.saturation,
        hue_shift      = args.hue,
        tone_method    = args.tone,
        denoise_sigma  = args.denoise,
        sharpen_amount = args.sharpen,
        zoom           = args.zoom,
        output_width   = args.width,
        output_height  = args.height,
        jpeg_quality   = args.quality,
        save_stages    = args.save_stages,
    )


if __name__ == "__main__":
    main()
