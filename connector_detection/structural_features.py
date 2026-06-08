from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


@dataclass(frozen=True)
class StructuralFeatureConfig:
    projection_dims: int = 128
    bright_threshold: float = 0.65
    edge_threshold: float = 0.12
    peak_threshold_std: float = 0.5
    peak_min_distance: int = 3


BASE_FEATURE_NAMES = [
    "aspect_ratio",
    "edge_density",
    "bright_ratio",
    "mean_intensity",
    "std_intensity",
    "peak_count",
    "peak_spacing_mean",
    "peak_spacing_std",
]


def structural_feature_names(projection_dims: int) -> list[str]:
    return BASE_FEATURE_NAMES + [f"projection_profile_{i:03d}" for i in range(projection_dims)]


def _resample_1d(values: np.ndarray, target_size: int) -> np.ndarray:
    if values.size == target_size:
        return values.astype(np.float32)
    source_x = np.linspace(0.0, 1.0, num=values.size)
    target_x = np.linspace(0.0, 1.0, num=target_size)
    return np.interp(target_x, source_x, values).astype(np.float32)


def _smooth_1d(values: np.ndarray, window_size: int = 5) -> np.ndarray:
    if values.size < window_size:
        return values
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(values, kernel, mode="same")


def _find_peaks(
    profile: np.ndarray,
    threshold_std: float,
    min_distance: int,
) -> np.ndarray:
    if profile.size < 3:
        return np.array([], dtype=np.int64)

    smoothed = _smooth_1d(profile, window_size=3)
    threshold = float(smoothed.mean() + threshold_std * smoothed.std())
    active = smoothed > threshold
    if not np.any(active):
        return np.array([], dtype=np.int64)

    padded = np.pad(active.astype(np.int8), (1, 1))
    starts = np.where(np.diff(padded) == 1)[0]
    ends = np.where(np.diff(padded) == -1)[0]
    candidates = np.array([(start + end - 1) // 2 for start, end in zip(starts, ends, strict=True)])

    selected: list[int] = []
    for candidate in candidates:
        if not selected or candidate - selected[-1] >= min_distance:
            selected.append(int(candidate))
        else:
            previous = selected[-1]
            merged = int(round((previous + candidate) / 2))
            selected[-1] = merged
    return np.array(selected, dtype=np.int64)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    edge = np.sqrt(gx * gx + gy * gy)
    max_value = float(edge.max())
    if max_value > 0:
        edge = edge / max_value
    return edge


def compute_structural_feature(
    image_path: Path,
    config: StructuralFeatureConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    with Image.open(image_path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("L")
        width, height = image.size
        gray = np.asarray(image, dtype=np.float32) / 255.0

    edge = _edge_map(gray)
    bright_mask = gray >= config.bright_threshold
    edge_mask = edge >= config.edge_threshold

    bright_projection = bright_mask.mean(axis=0).astype(np.float32)
    edge_projection = edge.mean(axis=0).astype(np.float32)
    projection = 0.5 * bright_projection + 0.5 * edge_projection
    projection = _resample_1d(projection, config.projection_dims)

    peaks = _find_peaks(
        projection,
        threshold_std=config.peak_threshold_std,
        min_distance=config.peak_min_distance,
    )
    peak_spacings = np.diff(peaks).astype(np.float32)
    peak_spacing_mean = float(peak_spacings.mean()) if peak_spacings.size else 0.0
    peak_spacing_std = float(peak_spacings.std()) if peak_spacings.size else 0.0

    base_values = np.array(
        [
            width / max(height, 1),
            float(edge_mask.mean()),
            float(bright_mask.mean()),
            float(gray.mean()),
            float(gray.std()),
            float(peaks.size),
            peak_spacing_mean,
            peak_spacing_std,
        ],
        dtype=np.float32,
    )
    feature = np.concatenate([base_values, projection.astype(np.float32)])
    summary = {
        "aspect_ratio": float(base_values[0]),
        "edge_density": float(base_values[1]),
        "bright_ratio": float(base_values[2]),
        "mean_intensity": float(base_values[3]),
        "std_intensity": float(base_values[4]),
        "peak_count": float(base_values[5]),
        "peak_spacing_mean": float(base_values[6]),
        "peak_spacing_std": float(base_values[7]),
    }
    return feature, summary


def compute_structural_features(
    image_paths: list[Path],
    config: StructuralFeatureConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    features = []
    summaries = []
    for image_path in tqdm(image_paths, desc="Structural features"):
        feature, summary = compute_structural_feature(image_path, config)
        features.append(feature)
        summaries.append({"image_path": str(image_path), **summary})
    return np.vstack(features), pd.DataFrame(summaries)
