from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "connector_detection_matplotlib"),
)
import matplotlib.pyplot as plt

@dataclass
class NearestCentroidModel:
    labels: list[str]
    centroids: np.ndarray
    distance_thresholds: dict[str, float]

    def predict(self, embeddings: np.ndarray) -> pd.DataFrame:
        distances = _pairwise_euclidean(embeddings, self.centroids)
        nearest_indices = distances.argmin(axis=1)
        nearest_distances = distances[np.arange(len(embeddings)), nearest_indices]
        predicted_labels = [self.labels[index] for index in nearest_indices]
        thresholds = np.array(
            [self.distance_thresholds[label] for label in predicted_labels],
            dtype=np.float32,
        )
        output_labels = [
            label if distance <= threshold else "unknown"
            for label, distance, threshold in zip(
                predicted_labels,
                nearest_distances,
                thresholds,
                strict=True,
            )
        ]
        sorted_distances = np.sort(distances, axis=1)
        margins = (
            sorted_distances[:, 1] - sorted_distances[:, 0]
            if distances.shape[1] > 1
            else np.zeros(len(embeddings), dtype=np.float32)
        )
        return pd.DataFrame(
            {
                "predicted_label": output_labels,
                "nearest_centroid_label": predicted_labels,
                "nearest_centroid_distance": nearest_distances,
                "nearest_margin": margins,
                "unknown_threshold": thresholds,
            }
        )


def _pairwise_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def infer_labels_from_parent(
    image_paths: list[Path],
    label_depth: int,
) -> list[str]:
    labels = []
    for image_path in image_paths:
        parent = image_path.parent
        parts = []
        for _ in range(label_depth):
            parts.append(parent.name)
            parent = parent.parent
        labels.append("/".join(reversed(parts)))
    return labels


def fit_nearest_centroids(
    embeddings_path: Path,
    manifest_path: Path,
    image_dir: Path,
    output_dir: Path,
    label_depth: int = 1,
    threshold_quantile: float = 0.995,
    random_state: int = 42,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(embeddings_path)
    manifest = pd.read_csv(manifest_path)
    image_paths = [Path(path) for path in manifest["image_path"]]

    root = image_dir.resolve()
    labels = []
    for image_path in image_paths:
        resolved = image_path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{image_path} is not under labeled image dir {image_dir}") from exc
        labels.append(infer_labels_from_parent([image_path], label_depth=label_depth)[0])

    label_series = pd.Series(labels, name="label")
    unique_labels = sorted(label_series.unique())
    if len(unique_labels) < 2:
        raise ValueError("Nearest-centroid training needs at least two labeled folders.")

    centroids = np.vstack(
        [embeddings[label_series == label].mean(axis=0) for label in unique_labels]
    )
    distances = _pairwise_euclidean(embeddings, centroids)
    nearest_indices = distances.argmin(axis=1)
    nearest_labels = [unique_labels[index] for index in nearest_indices]
    nearest_distances = distances[np.arange(len(embeddings)), nearest_indices]

    distance_thresholds = {}
    summary_rows = []
    for label_index, label in enumerate(unique_labels):
        mask = (label_series == label).to_numpy()
        own_distances = distances[mask, label_index]
        threshold = float(np.quantile(own_distances, threshold_quantile))
        distance_thresholds[label] = threshold
        summary_rows.append(
            {
                "label": label,
                "count": int(mask.sum()),
                "centroid_distance_mean": float(own_distances.mean()),
                "centroid_distance_std": float(own_distances.std()),
                "centroid_distance_p95": float(np.quantile(own_distances, 0.95)),
                "centroid_distance_threshold": threshold,
            }
        )

    model = NearestCentroidModel(
        labels=unique_labels,
        centroids=centroids,
        distance_thresholds=distance_thresholds,
    )
    model_path = output_dir / "nearest_centroid_model.joblib"
    joblib.dump(model, model_path)

    assignments = manifest.copy()
    assignments["label"] = labels
    assignments["nearest_centroid_label"] = nearest_labels
    assignments["nearest_centroid_distance"] = nearest_distances
    assignments["is_correct"] = assignments["label"] == assignments["nearest_centroid_label"]
    if distances.shape[1] > 1:
        sorted_distances = np.sort(distances, axis=1)
        assignments["nearest_margin"] = sorted_distances[:, 1] - sorted_distances[:, 0]
    else:
        assignments["nearest_margin"] = 0.0
    assignments_path = output_dir / "nearest_centroid_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    summary_path = output_dir / "nearest_centroid_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    plot_path = plot_nearest_centroid_umap(
        embeddings=embeddings,
        labels=labels,
        centroids=centroids,
        centroid_labels=unique_labels,
        output_path=output_dir / "nearest_centroid_umap.png",
        random_state=random_state,
    )
    export_label_montages(assignments, output_dir / "montage")
    return assignments_path, summary_path, plot_path


def assign_nearest_centroids(
    embeddings_path: Path,
    manifest_path: Path,
    model_path: Path,
    output_path: Path,
) -> Path:
    embeddings = np.load(embeddings_path)
    manifest = pd.read_csv(manifest_path)
    model: NearestCentroidModel = joblib.load(model_path)
    predictions = model.predict(embeddings)
    output = pd.concat([manifest, predictions], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def plot_nearest_centroid_umap(
    embeddings: np.ndarray,
    labels: list[str],
    centroids: np.ndarray,
    centroid_labels: list[str],
    output_path: Path,
    random_state: int,
) -> Path:
    import umap

    combined = np.vstack([embeddings, centroids])
    n_neighbors = min(15, max(2, combined.shape[0] - 1))
    points = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=random_state,
    ).fit_transform(combined)
    sample_points = points[: len(embeddings)]
    centroid_points = points[len(embeddings) :]

    label_codes, label_names = pd.factorize(np.asarray(labels))
    plt.figure(figsize=(11, 8))
    scatter = plt.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        c=label_codes,
        cmap="tab20",
        s=10,
        alpha=0.75,
    )
    for point, label in zip(centroid_points, centroid_labels, strict=True):
        plt.scatter(point[0], point[1], marker="*", s=260, c="black")
        plt.text(point[0], point[1], f" {label}", fontsize=9, weight="bold")
    colorbar = plt.colorbar(scatter)
    colorbar.set_ticks(range(len(label_names)))
    colorbar.set_ticklabels(label_names)
    plt.title("Nearest-centroid UMAP using DINOv2 + structural features")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def export_label_montages(
    assignments: pd.DataFrame,
    output_dir: Path,
    samples_per_label: int = 30,
    tile_size: tuple[int, int] = (160, 80),
    columns: int = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, group in assignments.groupby("label"):
        group = group.sort_values("nearest_centroid_distance").head(samples_per_label)
        rows = (len(group) + columns - 1) // columns
        montage = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
        draw = ImageDraw.Draw(montage)
        for index, image_path in enumerate(group["image_path"]):
            with Image.open(image_path) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")
                image.thumbnail(tile_size, Image.Resampling.BICUBIC)
                x = (index % columns) * tile_size[0] + (tile_size[0] - image.width) // 2
                y = (index // columns) * tile_size[1] + (tile_size[1] - image.height) // 2
                montage.paste(image, (x, y))
                draw.rectangle(
                    [
                        (index % columns) * tile_size[0],
                        (index // columns) * tile_size[1],
                        (index % columns + 1) * tile_size[0] - 1,
                        (index // columns + 1) * tile_size[1] - 1,
                    ],
                    outline=(220, 220, 220),
                )
        safe_label = "".join(char if char.isalnum() or char in "-_." else "_" for char in label)
        montage.save(output_dir / f"{safe_label}.jpg")
