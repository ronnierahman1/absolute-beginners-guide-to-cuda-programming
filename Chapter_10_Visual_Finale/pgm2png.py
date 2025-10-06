#!/usr/bin/env python3
"""
pgm2png.py — Convert PGM images to PNG.

Features:
    - Supports P2 (ASCII) and P5 (binary) PGM
    - Handles 8-bit and 16-bit grayscale; preserves 16-bit depth in PNG
    - Convert a single file, a list of files, or a directory (optionally recursive)
    - Custom output directory; optional overwrite
    - Quiet/verbose modes

Dependencies (either option works):
    - Option A (recommended): Pillow 10+  ->  pip install pillow
    - Option B (fallback):   imageio, numpy -> pip install imageio numpy

Usage:
    python pgm2png.py input.pgm [input2.pgm ...] [-o OUTDIR] [-r] [-f] [-q]
    See --help for details.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable, List

######################################################################
# Try Pillow first for broad format support; fall back to imageio+numpy.
######################################################################
try:
    # Try to import Pillow first, as it is the most robust and widely used image library for Python.
    from PIL import Image  # type: ignore
    _BACKEND = "pillow"
except Exception:  # pragma: no cover
    # If Pillow is not available, fall back to imageio and numpy.
    Image = None
    try:
        import imageio.v3 as iio  # type: ignore
        import numpy as np  # type: ignore
        _BACKEND = "imageio"
    except Exception as e:
        # If neither backend is available, print a clear error and re-raise.
        print("Error: You need either Pillow (`pip install pillow`) "
              "or imageio+numpy (`pip install imageio numpy`).")
        raise

PGM_EXTS = {".pgm"}  # Supported PGM file extensions

def find_inputs(paths: List[Path], recursive: bool) -> List[Path]:
    """
    Find all PGM files in the given list of files/directories.
    If a directory is given, search for .pgm files (recursively if requested).
    Returns a sorted, deduplicated list of Path objects.

    Args:
        paths: List of Path objects (files or directories) to search.
        recursive: If True, search directories recursively.

    Returns:
        List of Path objects for all found .pgm files.
    """
    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            if recursive:
                # Recursively search for .pgm files in all subdirectories
                files.extend([f for f in p.rglob("*") if f.suffix.lower() in PGM_EXTS])
            else:
                # Only search the top-level directory for .pgm files
                files.extend([f for f in p.glob("*") if f.suffix.lower() in PGM_EXTS])
        else:
            if p.suffix.lower() in PGM_EXTS:
                files.append(p)
    # Deduplicate and sort for stable output
    return sorted(set(files))

def out_path_for(src: Path, outdir: Path | None) -> Path:
    """
    Compute output PNG path for a given input PGM file.
    If outdir is given, place PNG there; else, alongside input.

    Args:
        src: Path to the input PGM file.
        outdir: Optional Path to output directory.

    Returns:
        Path to the output PNG file.
    """
    base = src.stem + ".png"
    return (outdir / base) if outdir else (src.with_suffix(".png"))

def convert_one_pillow(src: Path, dst: Path, overwrite: bool, verbose: bool) -> bool:
    """
    Convert a single PGM file to PNG using Pillow backend.
    Handles both 8-bit and 16-bit grayscale, preserving bit depth.

    Args:
        src: Path to input PGM file.
        dst: Path to output PNG file.
        overwrite: If True, overwrite existing PNGs.
        verbose: If True, print status messages.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    if dst.exists() and not overwrite:
        if verbose:
            print(f"[skip] {dst} (exists)")
        return False
    try:
        with Image.open(src) as im:
            # Pillow will open PGM as mode "L" (8-bit) or "I;16"/"I" for 16-bit.
            # Convert appropriately to keep bit depth.
            if im.mode == "L":  # 8-bit grayscale
                to_save = im
                save_params = {}
            else:
                # For 16-bit, ensure we save as 16-bit grayscale PNG
                if im.mode not in ("I;16", "I;16B", "I;16L", "I"):
                    im = im.convert("I")
                to_save = im
                save_params = {"bits": 16}
            # Ensure output directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            to_save.save(dst, format="PNG", **save_params)
        if verbose:
            print(f"[ok]   {src} -> {dst}  ({_BACKEND})")
        return True
    except Exception as e:
        # Print error and return False if conversion fails
        print(f"[fail] {src}: {e}")
        return False

def convert_one_imageio(src: Path, dst: Path, overwrite: bool, verbose: bool) -> bool:
    """
    Convert a single PGM file to PNG using imageio+numpy backend.
    Handles both 8-bit and 16-bit grayscale, preserving bit depth.

    Args:
        src: Path to input PGM file.
        dst: Path to output PNG file.
        overwrite: If True, overwrite existing PNGs.
        verbose: If True, print status messages.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    if dst.exists() and not overwrite:
        if verbose:
            print(f"[skip] {dst} (exists)")
        return False
    try:
        arr = iio.imread(src)  # dtype will reflect 8-bit (uint8) or 16-bit (uint16)
        # Ensure grayscale shape (H,W) or (H,W,1) -> (H,W)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        # imageio will preserve dtype to PNG automatically (uint8/uint16)
        dst.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(dst, arr)
        if verbose:
            bit = 16 if arr.dtype == getattr(np, "uint16", None) else 8
            print(f"[ok]   {src} -> {dst}  ({_BACKEND}, {bit}-bit)")
        return True
    except Exception as e:
        # Print error and return False if conversion fails
        print(f"[fail] {src}: {e}")
        return False

def convert_one(src: Path, dst: Path, overwrite: bool, verbose: bool) -> bool:
    """
    Convert a single PGM file to PNG using the selected backend.
    Chooses Pillow or imageio+numpy depending on what is available.

    Args:
        src: Path to input PGM file.
        dst: Path to output PNG file.
        overwrite: If True, overwrite existing PNGs.
        verbose: If True, print status messages.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    if _BACKEND == "pillow":
        return convert_one_pillow(src, dst, overwrite, verbose)
    else:
        return convert_one_imageio(src, dst, overwrite, verbose)

def main(argv: Iterable[str] | None = None) -> int:
    """
    Main entry point: parse arguments, find PGM files, convert to PNG.
    Returns 0 on full success, 2 if some files failed, 1 if none converted.

    Args:
        argv: Optional list of command-line arguments (for testing or scripting).

    Returns:
        int: Exit code (0=all success, 2=some failed, 1=none converted)
    """
    ap = argparse.ArgumentParser(
        description="Convert PGM images (P2/P5, 8/16-bit) to PNG."
    )
    ap.add_argument("inputs", nargs="+",
                    help="PGM file(s) or directory(ies) to convert.")
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
        print("No PGM files found.")
        return 1

    verbose = not args.quiet
    if verbose:
        print(f"Backend: {_BACKEND}")
        print(f"Found {len(files)} PGM file(s). Converting...")

    ok = 0
    for src in files:
        dst = out_path_for(src, args.outdir)
        if convert_one(src, dst, overwrite=args.force, verbose=verbose):
            ok += 1

    if verbose:
        print(f"Done. {ok}/{len(files)} converted.")
    # Return code meanings:
    #   0: all files converted successfully
    #   2: some files converted, some failed
    #   1: no files converted (all failed or none found)
    return 0 if ok == len(files) else (2 if ok > 0 else 1)

if __name__ == "__main__":
    # Run the main function if executed as a script
    sys.exit(main())
