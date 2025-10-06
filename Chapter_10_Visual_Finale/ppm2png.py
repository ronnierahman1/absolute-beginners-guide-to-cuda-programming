#!/usr/bin/env python3
"""
ppm2png.py — Convert PPM images (P3/P6, 8/16-bit RGB) to PNG.

Highlights:
    - Supports P3 (ASCII) and P6 (binary) PPM
    - Preserves 8-bit and 16-bit per-channel color when possible
    - Convert a single file, a list of files, or entire directories (optionally recursive)
    - Custom output directory; optional overwrite; quiet/verbose modes

Dependencies (either option works):
    - Option A (recommended for 16-bit RGB): imageio + numpy -> pip install imageio numpy
    - Option B (OK for common 8-bit files): Pillow -> pip install pillow
      (Note: Pillow may not preserve 16-bit RGB; if 16-bit is important, install imageio.)

Usage:
    python ppm2png.py input.ppm [input2.ppm ...] [-o OUTDIR] [-r] [-f] [-q]
    See --help for details.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable, List

PPM_EXTS = {".ppm"}  # Supported PPM file extensions

######################################################################
# Load backends (prefer imageio for 16-bit color fidelity)
######################################################################
_BACKEND = None
# Backend module placeholders; populated below if available.
iio = np = Image = None

# Prefer imageio + numpy because imageio preserves 16-bit RGB PNG output.
# Fall back to Pillow which is widely available but may downcast 16-bit RGB to 8-bit.
try:
    import imageio.v3 as iio  # type: ignore
    import numpy as np        # type: ignore
    _BACKEND = "imageio"
except Exception:
    try:
        # Pillow (PIL) provides basic PPM reading and PNG writing. Use when
        # imageio/numpy aren't installed. Note: Pillow may not preserve
        # 16-bit-per-channel RGB when saving to PNG.
        from PIL import Image  # type: ignore
        _BACKEND = "pillow"
    except Exception:
        # No supported backend available.
        pass

if _BACKEND is None:
    print("Error: Install either:\n"
          "  - imageio + numpy  -> pip install imageio numpy\n"
          "  - or Pillow        -> pip install pillow")
    # Exit with non-zero code since we cannot proceed without an image backend.
    sys.exit(1)


def find_inputs(paths: List[Path], recursive: bool) -> List[Path]:
    """
    Find all PPM files in the given list of files/directories.
    If a directory is given, search for .ppm files (recursively if requested).
    Returns a sorted, deduplicated list of Path objects.
    """
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            files.extend([f for f in iterator if f.suffix.lower() in PPM_EXTS])
        else:
            if p.suffix.lower() in PPM_EXTS:
                files.append(p)
    return sorted(set(files))


def out_path_for(src: Path, outdir: Path | None) -> Path:
    """
    Compute output PNG path for a given input PPM file.
    If outdir is given, place PNG there; else, alongside input.
    """
    base = src.stem + ".png"
    return (outdir / base) if outdir else src.with_suffix(".png")


def convert_one_imageio(src: Path, dst: Path, overwrite: bool, verbose: bool) -> bool:
    """
    Convert a single PPM file to PNG using imageio+numpy backend.
    Handles both 8-bit and 16-bit RGB, preserving bit depth.
    """
    if dst.exists() and not overwrite:
        if verbose:
            print(f"[skip] {dst} (exists)")
        return False
    try:
        # Read image into numpy array. imageio will choose dtype based on
        # file contents (uint8 for 8-bit PPM, uint16 for 16-bit PPM).
        arr = iio.imread(src)  # shape typically (H,W,3)

        # Normalize shape to (H,W,3) RGB. Some PPM variants or readers may
        # return (H,W) grayscale or (H,W,1) single-channel arrays; promote
        # those to RGB by duplication so PNG written is 3-channel.
        if arr.ndim == 2:
            # Malformed or grayscale PPM: replicate channels
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        # Ensure destination directory exists, then write preserving dtype.
        dst.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(dst, arr)  # preserves uint8/uint16 -> 8/16-bit PNG
        if verbose:
            bit = 16 if arr.dtype == getattr(np, "uint16", None) else 8
            print(f"[ok]   {src} -> {dst}  (imageio, {bit}-bit RGB)")
        return True
    except Exception as e:
        print(f"[fail] {src}: {e}")
        return False


def convert_one_pillow(src: Path, dst: Path, overwrite: bool, verbose: bool) -> bool:
    """
    Convert a single PPM file to PNG using Pillow backend.
    Note: Pillow may not preserve 16-bit RGB depth.
    """
    if dst.exists() and not overwrite:
        if verbose:
            print(f"[skip] {dst} (exists)")
        return False
    try:
        # Pillow's Image.open returns a file-like object; using context manager
        # ensures the file is closed after saving. Pillow often reads PPM as
        # 8-bit 'RGB' mode; if another mode is returned, convert to 'RGB'.
        with Image.open(src) as im:
            if im.mode != "RGB":
                # This will downcast 16-bit color to 8-bit if present.
                im = im.convert("RGB")
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="PNG")
        if verbose:
            print(f"[ok]   {src} -> {dst}  (pillow, RGB)")
        return True
    except Exception as e:
        print(f"[fail] {src}: {e}")
        return False


def convert_one(src: Path, dst: Path, overwrite: bool, verbose: bool) -> bool:
    """
    Convert a single PPM file to PNG using the selected backend.
    """
    if _BACKEND == "imageio":
        return convert_one_imageio(src, dst, overwrite, verbose)
    else:
        return convert_one_pillow(src, dst, overwrite, verbose)


def main(argv: Iterable[str] | None = None) -> int:
    """
    Main entry point: parse arguments, find PPM files, convert to PNG.
    Returns 0 on full success, 2 if some files failed, 1 if none converted.
    """
    ap = argparse.ArgumentParser(
        description="Convert PPM images (P3/P6, 8/16-bit) to PNG."
    )
    ap.add_argument("inputs", nargs="+",
                    help="PPM file(s) or directory(ies) to convert.")
    ap.add_argument("-o", "--outdir", type=Path, default=None,
                    help="Output directory for PNGs (default: alongside inputs).")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Recurse into directories.")
    ap.add_argument("-f", "--force", action="store_true",
                    help="Overwrite existing PNGs.")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Quiet mode (only errors).")
    args = ap.parse_args(argv)

    # Convert input arguments to Path objects
    inputs = [Path(p) for p in args.inputs]
    files = find_inputs(inputs, recursive=args.recursive)

    if not files:
        print("No PPM files found.")
        return 1

    verbose = not args.quiet
    if verbose:
        print(f"Backend: {_BACKEND}")
        print(f"Found {len(files)} PPM file(s). Converting...")

    ok = 0
    for src in files:
        dst = out_path_for(src, args.outdir)
        if convert_one(src, dst, overwrite=args.force, verbose=verbose):
            ok += 1

    if verbose:
        print(f"Done. {ok}/{len(files)} converted.")
    return 0 if ok == len(files) else (2 if ok > 0 else 1)


if __name__ == "__main__":
    # Run the main function if executed as a script
    sys.exit(main())
