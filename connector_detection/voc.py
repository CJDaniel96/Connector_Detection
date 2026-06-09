from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image, ImageOps

from connector_detection.images import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class VocBox:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2


@dataclass(frozen=True)
class VocAnnotation:
    xml_path: Path
    image_name: str
    image_width: int | None
    image_height: int | None
    boxes: list[VocBox]


def parse_voc_xml(xml_path: Path) -> VocAnnotation:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename") or f"{xml_path.stem}.jpg"

    size = root.find("size")
    image_width = int(size.findtext("width")) if size is not None else None
    image_height = int(size.findtext("height")) if size is not None else None

    boxes = []
    for obj in root.findall("object"):
        label = obj.findtext("name") or "object"
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        boxes.append(
            VocBox(
                label=label,
                xmin=int(float(bndbox.findtext("xmin", "0"))),
                ymin=int(float(bndbox.findtext("ymin", "0"))),
                xmax=int(float(bndbox.findtext("xmax", "0"))),
                ymax=int(float(bndbox.findtext("ymax", "0"))),
            )
        )

    return VocAnnotation(
        xml_path=xml_path,
        image_name=filename,
        image_width=image_width,
        image_height=image_height,
        boxes=boxes,
    )


def find_image_path(annotation: VocAnnotation, image_dir: Path) -> Path:
    candidates = [image_dir / annotation.image_name]
    candidates.extend(image_dir / f"{annotation.xml_path.stem}{ext}" for ext in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Image for {annotation.xml_path} not found under {image_dir}. "
        f"Tried filename={annotation.image_name} and matching XML stem."
    )


def classify_pin_band_orientation(box: VocBox, image_width: int, image_height: int) -> str:
    if box.width >= box.height:
        return "up" if box.center_y < image_height / 2 else "down"
    return "left" if box.center_x < image_width / 2 else "right"


def normalize_pin_band_to_up(crop: Image.Image, orientation: str) -> Image.Image:
    if orientation == "up":
        return crop
    if orientation == "down":
        return crop.transpose(Image.Transpose.ROTATE_180)
    if orientation == "left":
        return crop.transpose(Image.Transpose.ROTATE_270)
    if orientation == "right":
        return crop.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(f"Unsupported pin band orientation: {orientation}")


def rotate_image_orientation(image: Image.Image, source_orientation: str, target_orientation: str) -> Image.Image:
    orientations = ("up", "right", "down", "left")
    if source_orientation not in orientations:
        raise ValueError(f"Unsupported source orientation: {source_orientation}")
    if target_orientation not in orientations:
        raise ValueError(f"Unsupported target orientation: {target_orientation}")

    source_index = orientations.index(source_orientation)
    target_index = orientations.index(target_orientation)
    clockwise_turns = (target_index - source_index) % len(orientations)
    if clockwise_turns == 0:
        return image
    if clockwise_turns == 1:
        return image.transpose(Image.Transpose.ROTATE_270)
    if clockwise_turns == 2:
        return image.transpose(Image.Transpose.ROTATE_180)
    return image.transpose(Image.Transpose.ROTATE_90)


def rotate_images_from_voc_orientation(
    xml_dir: Path,
    image_dir: Path,
    output_dir: Path,
    label: str,
    target_orientation: str = "up",
    image_format: str = "png",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    xml_paths = sorted(xml_dir.rglob("*.xml"))
    if not xml_paths:
        raise ValueError(f"No XML files found under {xml_dir}")

    for xml_path in xml_paths:
        annotation = parse_voc_xml(xml_path)
        image_path = find_image_path(annotation, image_dir)

        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            image_width, image_height = image.size

            boxes = [box for box in annotation.boxes if box.label == label]
            for box_index, box in enumerate(boxes):
                orientation = classify_pin_band_orientation(box, image_width, image_height)
                rotated = rotate_image_orientation(image, orientation, target_orientation)
                output_name = (
                    f"{image_path.stem}_{box_index}_{box.label}"
                    f"_image_from-{orientation}_to-{target_orientation}.{image_format.lower()}"
                )
                rotated_path = output_dir / output_name
                rotated.save(rotated_path)

                rows.append(
                    {
                        "xml_path": str(xml_path),
                        "image_path": str(image_path),
                        "rotated_image_path": str(rotated_path),
                        "label": box.label,
                        "source_orientation": orientation,
                        "target_orientation": target_orientation,
                        "xmin": box.xmin,
                        "ymin": box.ymin,
                        "xmax": box.xmax,
                        "ymax": box.ymax,
                        "bbox_width": box.width,
                        "bbox_height": box.height,
                        "center_x": box.center_x,
                        "center_y": box.center_y,
                        "image_width": image_width,
                        "image_height": image_height,
                        "rotated_image_width": rotated.width,
                        "rotated_image_height": rotated.height,
                    }
                )

    if not rows:
        raise ValueError(f"No VOC objects with label {label} found under {xml_dir}")

    manifest_path = output_dir / "rotated_images_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def crop_pin_bands_from_voc(
    xml_dir: Path,
    image_dir: Path,
    output_dir: Path,
    label: str | None = None,
    padding: int = 0,
    image_format: str = "png",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    xml_paths = sorted(xml_dir.rglob("*.xml"))
    if not xml_paths:
        raise ValueError(f"No XML files found under {xml_dir}")

    for xml_path in xml_paths:
        annotation = parse_voc_xml(xml_path)
        image_path = find_image_path(annotation, image_dir)

        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            image_width, image_height = image.size

            boxes = [box for box in annotation.boxes if label is None or box.label == label]
            for box_index, box in enumerate(boxes):
                orientation = classify_pin_band_orientation(box, image_width, image_height)
                x1 = max(0, box.xmin - padding)
                y1 = max(0, box.ymin - padding)
                x2 = min(image_width, box.xmax + padding)
                y2 = min(image_height, box.ymax + padding)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = normalize_pin_band_to_up(image.crop((x1, y1, x2, y2)), orientation)
                output_name = (
                    f"{image_path.stem}_{box_index}_{box.label}"
                    f"_from-{orientation}_to-up.{image_format.lower()}"
                )
                crop_path = output_dir / output_name
                crop.save(crop_path)

                rows.append(
                    {
                        "xml_path": str(xml_path),
                        "image_path": str(image_path),
                        "crop_path": str(crop_path),
                        "label": box.label,
                        "source_orientation": orientation,
                        "normalized_orientation": "up",
                        "xmin": box.xmin,
                        "ymin": box.ymin,
                        "xmax": box.xmax,
                        "ymax": box.ymax,
                        "bbox_width": box.width,
                        "bbox_height": box.height,
                        "center_x": box.center_x,
                        "center_y": box.center_y,
                        "image_width": image_width,
                        "image_height": image_height,
                    }
                )

    manifest_path = output_dir / "pin_band_crops_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
