"""Pin localization in pin_row crops via NCC template matching.

A single-pin template is slid across the image; the resulting correlation
map (the "temperature map") is scanned for local maxima to locate each pin.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_correlation_map(image_gray: np.ndarray, template_gray: np.ndarray) -> np.ndarray:
    """Return the NCC correlation map (float32, values in [-1, 1]).
    Uses cv2.TM_CCOEFF_NORMED.
    """
    return cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)


def extract_template(image_gray: np.ndarray, pin_width_estimate: int) -> np.ndarray:
    """Auto-extract a single-pin template near the image center.

    Searches within one pin-width of the horizontal center for the window
    with the highest pixel variance — a high-variance region is more likely
    to land on a pin surface rather than on a gap between pins.
    """
    h, w = image_gray.shape[:2]
    actual_w = min(pin_width_estimate, w)
    cx = w // 2

    search_start = max(0, cx - actual_w)
    search_end = min(w - actual_w, cx + actual_w)

    if search_start >= search_end:
        x1 = max(0, cx - actual_w // 2)
        return image_gray[:, x1 : min(w, x1 + actual_w)].copy()

    best_x, best_var = search_start, -1.0
    for x in range(search_start, search_end + 1):
        v = float(np.var(image_gray[:, x : x + actual_w]))
        if v > best_var:
            best_var, best_x = v, x

    return image_gray[:, best_x : best_x + actual_w].copy()


def find_pin_peaks(
    corr_map: np.ndarray,
    template_w: int,
    template_h: int,
    threshold: float = 0.5,
    min_distance: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Return list of (x1, y1, x2, y2) bounding boxes for each detected pin.

    Algorithm:
    1. Collapse corr_map to a 1-D profile by taking max along the Y axis.
    2. Find local maxima above threshold (positions where profile[i] >= both neighbours).
    3. Suppress peaks closer than min_distance apart (keep the higher-score one).
    4. Each bounding box: x1=peak_x, x2=peak_x+template_w, y spans full image height.

    min_distance defaults to template_w // 2 if not provided.
    """
    if min_distance is None:
        min_distance = max(1, template_w // 2)

    map_h, map_w = corr_map.shape[:2]
    img_h = map_h + template_h - 1   # recover original image height
    img_w = map_w + template_w - 1   # recover original image width

    profile = corr_map.max(axis=0)   # 1-D horizontal correlation profile

    # Local maxima above threshold: must be >= both immediate neighbours
    candidates: list[tuple[int, float]] = []
    for i in range(map_w):
        if profile[i] < threshold:
            continue
        left_ok  = (i == 0)          or (profile[i] >= profile[i - 1])
        right_ok = (i == map_w - 1)  or (profile[i] >= profile[i + 1])
        if left_ok and right_ok:
            candidates.append((i, float(profile[i])))

    # NMS: among peaks closer than min_distance, keep the highest-score one
    candidates.sort(key=lambda p: p[1], reverse=True)
    kept: list[tuple[int, float]] = []
    for px, ps in candidates:
        if all(abs(px - kx) >= min_distance for kx, _ in kept):
            kept.append((px, ps))
    kept.sort(key=lambda p: p[0])

    boxes: list[tuple[int, int, int, int]] = []
    for px, _ in kept:
        x1 = px
        y1 = 0
        x2 = min(img_w, px + template_w)
        y2 = img_h
        if x2 > x1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def annotate_pins(
    image_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """Draw numbered bounding boxes on a copy of image_bgr and return it.

    Each box is labelled with its 1-based index (P1, P2, …).
    """
    out = image_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"P{i + 1}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = y1 + th + baseline + 2          # label inside the box, near top edge
        cv2.putText(out, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return out
