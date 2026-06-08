from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from connector_detection.features import build_combined_embeddings
from connector_detection.images import list_images

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "connector_detection_matplotlib"),
)
import matplotlib.pyplot as plt


GOOD_TOKENS = {"good", "ok", "normal", "pass"}
ANOMALY_TOKENS = {
    "abnormal",
    "anomaly",
    "bad",
    "defect",
    "dirty",
    "foreign",
    "missing",
    "ng",
    "shift",
}


@dataclass
class PatchCoreBank:
    label: str
    embeddings: np.ndarray
    threshold: float
    train_distance_mean: float
    train_distance_std: float
    train_count: int


@dataclass
class PatchCoreModel:
    banks: dict[str, PatchCoreBank]
    structural_scaler: StandardScaler
    structural_weight: float
    feature_config: dict[str, float | int | str]

    @property
    def labels(self) -> list[str]:
        return sorted(self.banks)

    def score_against_label(self, embeddings: np.ndarray, label: str) -> np.ndarray:
        if label not in self.banks:
            return np.full(len(embeddings), np.inf, dtype=np.float32)
        bank = self.banks[label]
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(bank.embeddings)
        distances, _ = nn.kneighbors(embeddings)
        return distances[:, 0]

    def nearest_labels(self, embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
        scores = []
        labels = self.labels
        for label in labels:
            scores.append(self.score_against_label(embeddings, label))
        score_matrix = np.vstack(scores).T
        nearest_indices = score_matrix.argmin(axis=1)
        return [labels[index] for index in nearest_indices], score_matrix[
            np.arange(len(embeddings)), nearest_indices
        ]


def infer_labels_from_root(
    image_paths: list[Path],
    root: Path,
    class_depth: int,
) -> list[str]:
    labels = []
    root = root.resolve()
    for image_path in image_paths:
        relative = image_path.resolve().relative_to(root)
        if len(relative.parts) <= class_depth:
            raise ValueError(
                f"{image_path} does not have enough path components for class_depth={class_depth}"
            )
        labels.append("/".join(relative.parts[:class_depth]))
    return labels


def infer_ground_truth_anomaly(
    image_path: Path,
    root: Path,
    class_depth: int,
) -> bool | None:
    relative = image_path.resolve().relative_to(root)
    tokens = {part.lower() for part in relative.parts[class_depth:-1]}
    if tokens & ANOMALY_TOKENS:
        return True
    if tokens & GOOD_TOKENS:
        return False
    return None


def train_patchcore_per_class(
    train_image_dir: Path,
    output_dir: Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    structural_weight: float,
    projection_profile_dims: int,
    bright_threshold: float,
    edge_threshold: float,
    peak_threshold_std: float,
    peak_min_distance: int,
    class_depth: int = 1,
    threshold_quantile: float = 0.995,
    validation_image_dir: Path | None = None,
    device: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = list_images(train_image_dir)
    if not train_paths:
        raise ValueError(f"No training images found under {train_image_dir}")

    train_feature_dir = output_dir / "features" / "train"
    train_embeddings, train_structural_summary, scaler = build_combined_embeddings(
        paths=train_paths,
        output_dir=train_feature_dir,
        model_name=model_name,
        image_size=image_size,
        batch_size=batch_size,
        structural_weight=structural_weight,
        projection_profile_dims=projection_profile_dims,
        bright_threshold=bright_threshold,
        edge_threshold=edge_threshold,
        peak_threshold_std=peak_threshold_std,
        peak_min_distance=peak_min_distance,
        device=device,
    )
    train_labels = infer_labels_from_root(train_paths, train_image_dir, class_depth)
    train_manifest = pd.DataFrame({"image_path": [str(path) for path in train_paths]})
    train_manifest = train_manifest.merge(train_structural_summary, on="image_path", how="left")
    train_manifest["label"] = train_labels
    train_manifest.to_csv(train_feature_dir / "manifest.csv", index=False)

    banks = {}
    train_rows = []
    for label in sorted(set(train_labels)):
        mask = np.asarray(train_labels) == label
        bank_embeddings = train_embeddings[mask]
        train_distances = _leave_one_out_distances(bank_embeddings)
        threshold = float(np.quantile(train_distances, threshold_quantile))
        banks[label] = PatchCoreBank(
            label=label,
            embeddings=bank_embeddings,
            threshold=threshold,
            train_distance_mean=float(train_distances.mean()),
            train_distance_std=float(train_distances.std()),
            train_count=int(mask.sum()),
        )
        label_paths = np.asarray(train_paths, dtype=object)[mask]
        for path, distance in zip(label_paths, train_distances, strict=True):
            train_rows.append(
                {
                    "image_path": str(path),
                    "label": label,
                    "patchcore_score": float(distance),
                    "threshold": threshold,
                    "predicted_anomaly": bool(distance > threshold),
                }
            )

    model = PatchCoreModel(
        banks=banks,
        structural_scaler=scaler,
        structural_weight=structural_weight,
        feature_config={
            "model_name": model_name,
            "image_size": image_size,
            "projection_profile_dims": projection_profile_dims,
            "bright_threshold": bright_threshold,
            "edge_threshold": edge_threshold,
            "peak_threshold_std": peak_threshold_std,
            "peak_min_distance": peak_min_distance,
            "class_depth": class_depth,
            "threshold_quantile": threshold_quantile,
        },
    )
    model_path = output_dir / "patchcore_models.joblib"
    joblib.dump(model, model_path)

    train_scores = pd.DataFrame(train_rows)
    train_scores.to_csv(output_dir / "train_patchcore_scores.csv", index=False)
    validation_scores = None
    if validation_image_dir is not None:
        validation_scores = validate_patchcore_per_class(
            model=model,
            validation_image_dir=validation_image_dir,
            output_dir=output_dir,
            model_name=model_name,
            image_size=image_size,
            batch_size=batch_size,
            projection_profile_dims=projection_profile_dims,
            bright_threshold=bright_threshold,
            edge_threshold=edge_threshold,
            peak_threshold_std=peak_threshold_std,
            peak_min_distance=peak_min_distance,
            class_depth=class_depth,
            device=device,
        )

    report_path = write_patchcore_report(
        model=model,
        train_scores=train_scores,
        validation_scores=validation_scores,
        output_dir=output_dir,
    )
    plot_score_histograms(train_scores, validation_scores, output_dir / "plots")
    export_anomaly_montages(train_scores, output_dir / "montage" / "train")
    if validation_scores is not None:
        export_anomaly_montages(validation_scores, output_dir / "montage" / "validation")
    return model_path, report_path


def validate_patchcore_per_class(
    model: PatchCoreModel,
    validation_image_dir: Path,
    output_dir: Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    projection_profile_dims: int,
    bright_threshold: float,
    edge_threshold: float,
    peak_threshold_std: float,
    peak_min_distance: int,
    class_depth: int = 1,
    device: str | None = None,
) -> pd.DataFrame:
    validation_paths = list_images(validation_image_dir)
    if not validation_paths:
        raise ValueError(f"No validation images found under {validation_image_dir}")

    validation_feature_dir = output_dir / "features" / "validation"
    validation_embeddings, structural_summary, _ = build_combined_embeddings(
        paths=validation_paths,
        output_dir=validation_feature_dir,
        model_name=model_name,
        image_size=image_size,
        batch_size=batch_size,
        structural_weight=model.structural_weight,
        projection_profile_dims=projection_profile_dims,
        bright_threshold=bright_threshold,
        edge_threshold=edge_threshold,
        peak_threshold_std=peak_threshold_std,
        peak_min_distance=peak_min_distance,
        structural_scaler=model.structural_scaler,
        device=device,
    )

    expected_labels = infer_labels_from_root(validation_paths, validation_image_dir, class_depth)
    nearest_labels, nearest_scores = model.nearest_labels(validation_embeddings)
    expected_scores = []
    thresholds = []
    predicted_anomalies = []
    ground_truth = []
    for embedding, expected_label, image_path in zip(
        validation_embeddings,
        expected_labels,
        validation_paths,
        strict=True,
    ):
        if expected_label in model.banks:
            score = float(model.score_against_label(embedding.reshape(1, -1), expected_label)[0])
            threshold = model.banks[expected_label].threshold
        else:
            score = float("inf")
            threshold = float("nan")
        expected_scores.append(score)
        thresholds.append(threshold)
        predicted_anomalies.append(bool(score > threshold))
        ground_truth.append(infer_ground_truth_anomaly(image_path, validation_image_dir, class_depth))

    scores = pd.DataFrame(
        {
            "image_path": [str(path) for path in validation_paths],
            "label": expected_labels,
            "nearest_label": nearest_labels,
            "nearest_score": nearest_scores,
            "patchcore_score": expected_scores,
            "threshold": thresholds,
            "predicted_anomaly": predicted_anomalies,
            "ground_truth_anomaly": ground_truth,
        }
    )
    scores = scores.merge(structural_summary, on="image_path", how="left")
    scores.to_csv(output_dir / "validation_patchcore_scores.csv", index=False)
    return scores


def load_and_validate_patchcore(
    model_path: Path,
    validation_image_dir: Path,
    output_dir: Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    projection_profile_dims: int,
    bright_threshold: float,
    edge_threshold: float,
    peak_threshold_std: float,
    peak_min_distance: int,
    class_depth: int = 1,
    device: str | None = None,
) -> Path:
    model: PatchCoreModel = joblib.load(model_path)
    validation_scores = validate_patchcore_per_class(
        model=model,
        validation_image_dir=validation_image_dir,
        output_dir=output_dir,
        model_name=model_name,
        image_size=image_size,
        batch_size=batch_size,
        projection_profile_dims=projection_profile_dims,
        bright_threshold=bright_threshold,
        edge_threshold=edge_threshold,
        peak_threshold_std=peak_threshold_std,
        peak_min_distance=peak_min_distance,
        class_depth=class_depth,
        device=device,
    )
    report_path = write_patchcore_report(
        model=model,
        train_scores=None,
        validation_scores=validation_scores,
        output_dir=output_dir,
    )
    plot_score_histograms(None, validation_scores, output_dir / "plots")
    export_anomaly_montages(validation_scores, output_dir / "montage" / "validation")
    return report_path


def _leave_one_out_distances(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    return distances[:, 1]


def write_patchcore_report(
    model: PatchCoreModel,
    train_scores: pd.DataFrame | None,
    validation_scores: pd.DataFrame | None,
    output_dir: Path,
) -> Path:
    lines = ["# PatchCore Report", ""]
    lines.append("## Per-Class Banks")
    lines.append("")
    lines.append(
        "| label | train_count | threshold | train_distance_mean | train_distance_std |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label in model.labels:
        bank = model.banks[label]
        lines.append(
            f"| {label} | {bank.train_count} | {bank.threshold:.6f} | "
            f"{bank.train_distance_mean:.6f} | {bank.train_distance_std:.6f} |"
        )

    if train_scores is not None:
        lines.extend(["", "## Train Scores", ""])
        lines.append(f"- Images: {len(train_scores)}")
        lines.append(f"- Flagged as anomaly: {int(train_scores['predicted_anomaly'].sum())}")

    if validation_scores is not None:
        lines.extend(["", "## Validation", ""])
        lines.append(f"- Images: {len(validation_scores)}")
        lines.append(
            f"- Predicted anomaly: {int(validation_scores['predicted_anomaly'].sum())}"
        )
        metrics = _classification_metrics(validation_scores)
        if metrics:
            lines.append(f"- Accuracy: {metrics['accuracy']:.4f}")
            lines.append(f"- Precision: {metrics['precision']:.4f}")
            lines.append(f"- Recall: {metrics['recall']:.4f}")
            lines.append(f"- F1: {metrics['f1']:.4f}")

        lines.extend(["", "### Validation By Class", ""])
        lines.append("| label | count | predicted_anomaly | score_mean | score_p95 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for label, group in validation_scores.groupby("label"):
            lines.append(
                f"| {label} | {len(group)} | {int(group['predicted_anomaly'].sum())} | "
                f"{group['patchcore_score'].mean():.6f} | "
                f"{group['patchcore_score'].quantile(0.95):.6f} |"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `patchcore_models.joblib`: per-class feature banks and thresholds",
            "- `train_patchcore_scores.csv`: training leave-one-out scores",
            "- `validation_patchcore_scores.csv`: validation scores, if validation data was provided",
            "- `plots/*.png`: score histograms with thresholds",
            "- `montage/**.jpg`: highest-score examples per class",
        ]
    )
    report_path = output_dir / "patchcore_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _classification_metrics(scores: pd.DataFrame) -> dict[str, float]:
    labeled = scores.dropna(subset=["ground_truth_anomaly"])
    if labeled.empty:
        return {}
    y_true = labeled["ground_truth_anomaly"].astype(bool)
    y_pred = labeled["predicted_anomaly"].astype(bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(labeled), 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_score_histograms(
    train_scores: pd.DataFrame | None,
    validation_scores: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = set()
    if train_scores is not None:
        labels.update(train_scores["label"].unique())
    if validation_scores is not None:
        labels.update(validation_scores["label"].unique())

    for label in sorted(labels):
        plt.figure(figsize=(8, 5))
        threshold = None
        if train_scores is not None:
            train_group = train_scores[train_scores["label"] == label]
            if not train_group.empty:
                threshold = float(train_group["threshold"].iloc[0])
                plt.hist(
                    train_group["patchcore_score"],
                    bins=30,
                    alpha=0.55,
                    label="train",
                )
        if validation_scores is not None:
            val_group = validation_scores[validation_scores["label"] == label]
            if not val_group.empty:
                threshold = float(val_group["threshold"].iloc[0])
                plt.hist(
                    val_group["patchcore_score"].replace([np.inf], np.nan).dropna(),
                    bins=30,
                    alpha=0.55,
                    label="validation",
                )
        if threshold is not None and np.isfinite(threshold):
            plt.axvline(threshold, color="red", linestyle="--", label="threshold")
        plt.title(f"PatchCore score distribution: {label}")
        plt.xlabel("nearest bank distance")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        safe_label = _safe_filename(label)
        plt.savefig(output_dir / f"{safe_label}_histogram.png", dpi=160)
        plt.close()


def export_anomaly_montages(
    scores: pd.DataFrame,
    output_dir: Path,
    samples_per_label: int = 30,
    tile_size: tuple[int, int] = (180, 90),
    columns: int = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, group in scores.groupby("label"):
        group = group.sort_values("patchcore_score", ascending=False).head(samples_per_label)
        rows = max(1, (len(group) + columns - 1) // columns)
        montage = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
        draw = ImageDraw.Draw(montage)
        for index, row in enumerate(group.itertuples(index=False)):
            with Image.open(row.image_path) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")
                image.thumbnail(tile_size, Image.Resampling.BICUBIC)
                tile_x = (index % columns) * tile_size[0]
                tile_y = (index // columns) * tile_size[1]
                x = tile_x + (tile_size[0] - image.width) // 2
                y = tile_y + (tile_size[1] - image.height) // 2
                montage.paste(image, (x, y))
                outline = (220, 0, 0) if row.predicted_anomaly else (220, 220, 220)
                draw.rectangle(
                    [tile_x, tile_y, tile_x + tile_size[0] - 1, tile_y + tile_size[1] - 1],
                    outline=outline,
                    width=2,
                )
                draw.text(
                    (tile_x + 4, tile_y + 4),
                    f"{row.patchcore_score:.3f}",
                    fill=(0, 0, 0),
                )
        montage.save(output_dir / f"{_safe_filename(label)}_top_scores.jpg")


def _safe_filename(label: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in label)
