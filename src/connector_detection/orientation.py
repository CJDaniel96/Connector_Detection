"""Connector orientation normalization using YOLO #2 part detections.

Standard orientation: pin row on top, metal body on bottom.
Connector is landscape (width > height) in standard orientation.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

PIN_ROW_CLASS = 0
METAL_BODY_CLASS = 1


def _bbox_center(xyxy: np.ndarray) -> tuple[float, float]:
    return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)


def detect_parts(
    image: np.ndarray,
    model: YOLO,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run YOLO #2 and return (pin_row_xyxy, metal_body_xyxy).

    Returns None for each component if not detected.
    Uses highest-confidence detection per class.
    """
    results = model(image, verbose=False)
    pin_box: np.ndarray | None = None
    body_box: np.ndarray | None = None
    pin_conf = body_conf = -1.0

    if results:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            if cls == PIN_ROW_CLASS and conf > pin_conf:
                pin_box, pin_conf = xyxy, conf
            elif cls == METAL_BODY_CLASS and conf > body_conf:
                body_box, body_conf = xyxy, conf

    return pin_box, body_box


def determine_rotation(
    pin_center: tuple[float, float],
    body_center: tuple[float, float],
) -> int:
    """Return CCW rotation degrees (0 / 90 / 180 / 270) to reach standard orientation.

    Image coordinate convention: Y increases downward.
    Standard: pin_center_y < body_center_y (pin is above body).

    Decision table:
        dy < 0  and  |dy| >= |dx|  →  pins above  → 0° (no rotation)
        dy > 0  and  |dy| >= |dx|  →  pins below  → 180°
        dx > 0  and  |dx| >  |dy|  →  pins right  → 90° CCW
        dx < 0  and  |dx| >  |dy|  →  pins left   → 270° CCW (= 90° CW)
    """
    dx = pin_center[0] - body_center[0]
    dy = pin_center[1] - body_center[1]
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 180
    return 90 if dx > 0 else 270


def rotate_image(image: np.ndarray, angle_ccw: int) -> np.ndarray:
    """Rotate image counter-clockwise by angle_ccw degrees (0 / 90 / 180 / 270)."""
    if angle_ccw == 0:
        return image
    if angle_ccw == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle_ccw == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle_ccw == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"angle_ccw must be 0/90/180/270, got {angle_ccw}")


def _safe_crop(image: np.ndarray, xyxy: np.ndarray | None) -> np.ndarray | None:
    if xyxy is None:
        return None
    h, w = image.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(w, int(xyxy[2]))
    y2 = min(h, int(xyxy[3]))
    return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def normalize_connector(
    image: np.ndarray,
    model: YOLO,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Normalize connector to standard orientation and return part crops.

    Returns:
        (pin_crop, body_crop, rotation_angle_ccw)
        pin_crop / body_crop are None if the part was not detected.
        rotation_angle_ccw is the CCW degrees applied to the image.

    Algorithm:
        1. Detect pin_row and metal_body in original image.
        2. Determine rotation based on relative center positions.
        3. Rotate image to standard orientation.
        4. Re-detect parts in rotated image for accurate crop coordinates.
    """
    pin_box, body_box = detect_parts(image, model)
    if pin_box is None or body_box is None:
        return None, None, 0

    angle = determine_rotation(_bbox_center(pin_box), _bbox_center(body_box))
    rotated = rotate_image(image, angle)

    pin_box_rot, body_box_rot = detect_parts(rotated, model)

    return (
        _safe_crop(rotated, pin_box_rot),
        _safe_crop(rotated, body_box_rot),
        angle,
    )
