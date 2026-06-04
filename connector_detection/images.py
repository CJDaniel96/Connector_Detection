from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_with_padding(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def copy_review_images(paths: Iterable[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = output_dir / path.name
        if target.exists():
            target = output_dir / f"{path.stem}_{abs(hash(path))}{path.suffix}"
        with Image.open(path) as image:
            ImageOps.exif_transpose(image).save(target)
