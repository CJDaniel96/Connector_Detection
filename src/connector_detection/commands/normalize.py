"""Batch-normalize connector orientations and produce PatchCore-ready crops.

Usage (as script):
    uv run scripts/normalize_orientation.py <input_dir> --yolo-model models/yolo_parts.pt \\
        --split train --label good

Usage (as installed command):
    normalize-orientation <input_dir> --yolo-model models/yolo_parts.pt --split train --label good
"""

from __future__ import annotations

from pathlib import Path

import cv2
import typer
from ultralytics import YOLO

from connector_detection.orientation import normalize_connector

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@app.command()
def main(
    input_dir: Path = typer.Argument(..., help="Directory of connector images (cropped by YOLO #1, classified by DINOv2)"),
    yolo_model: Path = typer.Option(..., "--yolo-model", "-m", help="Path to YOLO #2 weights (.pt) — detects pin_row and metal_body"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", "-d", help="Output base data directory"),
    split: str = typer.Option("train", "--split", "-s", help="Dataset split: train | test"),
    label: str = typer.Option("good", "--label", "-l", help="Image label: good | bad"),
    output_suffix: str = typer.Option(".jpg", "--suffix", help="Output image file suffix"),
    conf_threshold: float = typer.Option(0.25, "--conf", help="YOLO detection confidence threshold"),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip images already present in output directory"),
) -> None:
    """Detect pin_row and metal_body, correct orientation, and save crops for PatchCore training."""
    if split not in ("train", "test"):
        typer.echo("Error: --split must be 'train' or 'test'", err=True)
        raise typer.Exit(1)
    if label not in ("good", "bad"):
        typer.echo("Error: --label must be 'good' or 'bad'", err=True)
        raise typer.Exit(1)
    if not input_dir.is_dir():
        typer.echo(f"Error: input directory not found: {input_dir}", err=True)
        raise typer.Exit(1)
    if not yolo_model.exists():
        typer.echo(f"Error: YOLO model not found: {yolo_model}", err=True)
        raise typer.Exit(1)

    pin_out = data_dir / "pin_row" / split / label
    body_out = data_dir / "metal_body" / split / label
    pin_out.mkdir(parents=True, exist_ok=True)
    body_out.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading YOLO #2 model: {yolo_model}")
    model = YOLO(str(yolo_model))
    model.conf = conf_threshold

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        typer.echo(f"No images found in {input_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(images)} images  |  split={split}  |  label={label}")
    typer.echo(f"  pin_row   → {pin_out}")
    typer.echo(f"  metal_body → {body_out}\n")

    processed = skipped = failed = 0
    rotation_counts: dict[int, int] = {0: 0, 90: 0, 180: 0, 270: 0}

    for img_path in typer.progressbar(images, label="Processing"):
        stem = img_path.stem
        pin_dest = pin_out / f"{stem}{output_suffix}"
        body_dest = body_out / f"{stem}{output_suffix}"

        if skip_existing and pin_dest.exists() and body_dest.exists():
            skipped += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            typer.echo(f"\n  [WARN] Cannot read: {img_path.name}", err=True)
            failed += 1
            continue

        pin_crop, body_crop, angle = normalize_connector(image, model)

        if pin_crop is None and body_crop is None:
            typer.echo(f"\n  [SKIP] No parts detected: {img_path.name}", err=True)
            failed += 1
            continue

        if pin_crop is not None:
            cv2.imwrite(str(pin_dest), pin_crop)
        else:
            typer.echo(f"\n  [WARN] pin_row not detected: {img_path.name}", err=True)

        if body_crop is not None:
            cv2.imwrite(str(body_dest), body_crop)
        else:
            typer.echo(f"\n  [WARN] metal_body not detected: {img_path.name}", err=True)

        rotation_counts[angle] = rotation_counts.get(angle, 0) + 1
        processed += 1

    _LABELS = {0: "standard", 90: "pins-right→CCW90°", 180: "inverted", 270: "pins-left→CW90°"}
    typer.echo(f"\n{'─' * 52}")
    typer.echo(f"Processed : {processed}")
    typer.echo(f"Skipped   : {skipped}")
    typer.echo(f"Failed    : {failed}")
    typer.echo("\nRotation distribution applied:")
    for angle, count in sorted(rotation_counts.items()):
        typer.echo(f"  {angle:3d}° {_LABELS[angle]}: {count}")
