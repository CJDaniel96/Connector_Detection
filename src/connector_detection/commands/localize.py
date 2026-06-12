"""Batch-localize individual pins in pin_row crops using NCC template matching.

Primary workflow — mark the first pin, find the rest automatically:
    The user identifies the first pin's bounding box in a reference image
    (x1,y1,x2,y2) and passes it via --first-pin.  The command crops that
    region from every input image as the NCC template, then slides it across
    the full image to locate all pins.

Usage (as script):
    # Primary: specify first-pin bounding box, let NCC find the rest
    uv run scripts/localize_pins.py data/pin_row/train/good/ \\
        --first-pin 4,0,32,48 \\
        --output-dir results/pin_localization/

    # Alternative: provide a pre-cropped single-pin image as template
    uv run scripts/localize_pins.py data/pin_row/train/good/ \\
        --template templates/single_pin.jpg \\
        --output-dir results/pin_localization/

Usage (as installed command):
    localize-pins data/pin_row/train/good/ --first-pin 4,0,32,48 --save-map
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import typer

from connector_detection.pin_localization import (
    annotate_pins,
    compute_correlation_map,
    extract_template,
    find_pin_peaks,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _parse_first_pin(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


@app.command()
def main(
    input_dir: Path = typer.Argument(..., help="Directory of pin_row crop images"),
    output_dir: Path = typer.Option(Path("results/pin_localization"), "--output-dir", "-o", help="Output directory"),
    first_pin: str = typer.Option(
        None, "--first-pin",
        help="Bounding box of the first pin within the image: x1,y1,x2,y2 (pixels). "
             "Crops this region from each input image as the NCC template.",
    ),
    template_path: Path = typer.Option(
        None, "--template", "-t",
        help="Pre-cropped single-pin image file to use as template. "
             "Takes effect only when --first-pin is not provided.",
    ),
    pin_width: int = typer.Option(
        30, "--pin-width", "-w",
        help="Fallback: estimated pin width in pixels for auto-template extraction "
             "(used only when neither --first-pin nor --template is provided).",
    ),
    threshold: float = typer.Option(0.5, "--threshold", help="NCC correlation threshold for peak detection (0–1)"),
    min_distance: int = typer.Option(
        0, "--min-distance",
        help="Minimum pixel distance between two detected peaks. "
             "0 = auto (half the template width).",
    ),
    save_map: bool = typer.Option(False, "--save-map", help="Save the NCC heat map alongside each annotated image"),
    save_crops: bool = typer.Option(False, "--save-crops", help="Save individual pin crops to output_dir/crops/<stem>/pin_N.jpg"),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip images whose annotated output already exists"),
) -> None:
    """Locate each individual pin in pin_row crops using NCC template matching.

    Recommended: pass --first-pin x1,y1,x2,y2 to mark the first pin, then
    the command uses that crop as the template to find all remaining pins.
    """
    if not input_dir.is_dir():
        typer.echo(f"Error: input directory not found: {input_dir}", err=True)
        raise typer.Exit(1)

    # --- Parse --first-pin ---------------------------------------------------
    first_pin_box: tuple[int, int, int, int] | None = None
    if first_pin is not None:
        try:
            first_pin_box = _parse_first_pin(first_pin)
        except (ValueError, TypeError):
            typer.echo("Error: --first-pin must be four comma-separated integers: x1,y1,x2,y2", err=True)
            raise typer.Exit(1)
        fx1, fy1, fx2, fy2 = first_pin_box
        if fx2 <= fx1 or fy2 <= fy1:
            typer.echo("Error: --first-pin x2 must be > x1 and y2 must be > y1", err=True)
            raise typer.Exit(1)
        typer.echo(f"Template source  : first pin at ({fx1},{fy1})→({fx2},{fy2}), cropped per image")

    # --- Load shared template from file (only when --first-pin not given) ----
    shared_template_gray: np.ndarray | None = None
    if first_pin_box is None and template_path is not None:
        if not template_path.exists():
            typer.echo(f"Error: template not found: {template_path}", err=True)
            raise typer.Exit(1)
        tmpl = cv2.imread(str(template_path))
        if tmpl is None:
            typer.echo(f"Error: cannot read template: {template_path}", err=True)
            raise typer.Exit(1)
        shared_template_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        typer.echo(f"Template source  : file {template_path}  ({shared_template_gray.shape[1]}×{shared_template_gray.shape[0]})")

    if first_pin_box is None and shared_template_gray is None:
        typer.echo(f"Template source  : auto-extract (pin-width={pin_width}px) — consider using --first-pin for better accuracy")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        typer.echo(f"No images found in {input_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(images)} images  |  threshold={threshold}")
    typer.echo(f"Output → {output_dir}\n")

    processed = skipped = failed = 0
    total_pins = 0
    rows: list[dict[str, object]] = []
    eff_min_dist = None if min_distance == 0 else min_distance

    for img_path in typer.progressbar(images, label="Localizing"):
        stem = img_path.stem
        out_path = output_dir / f"{stem}_pins.jpg"

        if skip_existing and out_path.exists():
            skipped += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            typer.echo(f"\n  [WARN] Cannot read: {img_path.name}", err=True)
            failed += 1
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape[:2]

        # Resolve template for this image
        if first_pin_box is not None:
            fx1, fy1, fx2, fy2 = first_pin_box
            if fx2 > img_w or fy2 > img_h:
                typer.echo(f"\n  [WARN] --first-pin exceeds image bounds {img_w}×{img_h}, skipping: {img_path.name}", err=True)
                failed += 1
                continue
            tmpl_gray = gray[fy1:fy2, fx1:fx2]
        elif shared_template_gray is not None:
            tmpl_gray = shared_template_gray
        else:
            tmpl_gray = extract_template(gray, pin_width)

        tmpl_h, tmpl_w = tmpl_gray.shape[:2]
        if tmpl_w > img_w or tmpl_h > img_h:
            typer.echo(f"\n  [WARN] Template larger than image, skipping: {img_path.name}", err=True)
            failed += 1
            continue

        corr_map = compute_correlation_map(gray, tmpl_gray)
        boxes = find_pin_peaks(corr_map, tmpl_w, tmpl_h, threshold, eff_min_dist)

        annotated = annotate_pins(bgr, boxes)
        cv2.imwrite(str(out_path), annotated)

        if save_map:
            norm = ((corr_map + 1) / 2 * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            cv2.imwrite(str(output_dir / f"{stem}_corr.png"), heatmap)

        if save_crops and boxes:
            crops_dir = output_dir / "crops" / stem
            crops_dir.mkdir(parents=True, exist_ok=True)
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                crop = bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    cv2.imwrite(str(crops_dir / f"pin_{i + 1}.jpg"), crop)

        total_pins += len(boxes)
        rows.append({"filename": stem, "pin_count": len(boxes)})
        processed += 1

    csv_path = output_dir / "pin_counts.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "pin_count"])
        writer.writeheader()
        writer.writerows(rows)

    avg = total_pins / processed if processed else 0.0
    typer.echo(f"\n{'─' * 52}")
    typer.echo(f"Processed : {processed}")
    typer.echo(f"Skipped   : {skipped}")
    typer.echo(f"Failed    : {failed}")
    typer.echo(f"Avg pins  : {avg:.1f}")
    typer.echo(f"CSV       : {csv_path}")
