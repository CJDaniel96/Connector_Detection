from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from connector_detection.features import build_combined_embeddings
from connector_detection.images import list_images
from connector_detection.patchcore import (
    ANOMALY_TOKENS,
    GOOD_TOKENS,
    infer_labels_from_root,
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "connector_detection_matplotlib"),
)
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DinoBankConfig:
    dinov2_model: str = "facebook/dinov2-small"
    image_size: int = 224
    batch_size: int = 16
    structural_weight: float = 2.0
    projection_profile_dims: int = 128
    bright_threshold: float = 0.65
    edge_threshold: float = 0.12
    peak_threshold_std: float = 0.5
    peak_min_distance: int = 3
    structural_image_size: int | tuple[int, int] | None = None
    pca_components: int = 50
    threshold_quantile: float = 0.995
    histogram_bins: int = 30
    montage_samples: int = 30
    random_state: int = 42


@dataclass
class DinoBankModel:
    config: DinoBankConfig
    class_depth: int
    labels: list[str]
    banks: dict[str, np.ndarray]
    thresholds: dict[str, float]
    structural_scaler: StandardScaler
    feature_scaler: StandardScaler
    pca: PCA | None


def train_dinobank(
    train_image_dir: Path,
    output_dir: Path,
    config: DinoBankConfig,
    class_depth: int = 1,
    validation_image_dir: Path | None = None,
    class_labels: list[str] | None = None,
    device: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = _filter_class_labels(
        _normal_training_paths(train_image_dir, class_depth),
        train_image_dir,
        class_depth,
        class_labels,
    )
    if not train_paths:
        raise ValueError(f"No normal training images found under {train_image_dir}")

    embeddings, manifest, structural_scaler = _extract_raw_embeddings(
        paths=train_paths,
        output_dir=output_dir / "train_features",
        config=config,
        structural_scaler=None,
        device=device,
    )
    labels = infer_labels_from_root(train_paths, train_image_dir, class_depth)
    manifest["label"] = labels

    feature_scaler = StandardScaler()
    scaled = feature_scaler.fit_transform(embeddings)
    pca = _fit_pca(scaled, config)
    bank_vectors = pca.fit_transform(scaled) if pca is not None else scaled

    banks: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    summary_rows: list[dict[str, Any]] = []
    for label in sorted(set(labels)):
        mask = np.asarray(labels) == label
        bank = bank_vectors[mask]
        banks[label] = bank
        loo_distances = _leave_one_out_distances(bank)
        threshold = float(np.quantile(loo_distances, config.threshold_quantile))
        thresholds[label] = threshold
        summary_rows.append(
            {
                "label": label,
                "count": int(mask.sum()),
                "threshold": threshold,
                "train_distance_mean": float(loo_distances.mean()),
                "train_distance_std": float(loo_distances.std()),
                "train_distance_p95": float(np.quantile(loo_distances, 0.95)),
            }
        )

    model = DinoBankModel(
        config=config,
        class_depth=class_depth,
        labels=sorted(banks),
        banks=banks,
        thresholds=thresholds,
        structural_scaler=structural_scaler,
        feature_scaler=feature_scaler,
        pca=pca,
    )
    model_path = output_dir / "dinobank_model.joblib"
    joblib.dump(model, model_path)

    train_scores = _score_known_labels(
        image_paths=train_paths,
        labels=labels,
        vectors=bank_vectors,
        model=model,
        ground_truth=None,
    )
    train_scores.to_csv(output_dir / "dinobank_train_scores.csv", index=False)
    manifest.to_csv(output_dir / "train_manifest.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "dinobank_summary.csv", index=False)

    validation_path = None
    if validation_image_dir is not None:
        validation_path = validate_dinobank(
            model_path=model_path,
            validation_image_dir=validation_image_dir,
            output_dir=output_dir / "validation",
            class_depth=None,
            class_labels=class_labels,
            device=device,
        )

    report_path = _write_report(
        output_dir=output_dir,
        model_path=model_path,
        summary=pd.DataFrame(summary_rows),
        validation_report=validation_path,
    )
    return model_path, report_path


def validate_dinobank(
    model_path: Path,
    validation_image_dir: Path,
    output_dir: Path,
    class_depth: int | None = None,
    class_labels: list[str] | None = None,
    device: str | None = None,
) -> Path:
    model: DinoBankModel = joblib.load(model_path)
    resolved_class_depth = class_depth or model.class_depth
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = sorted(set(class_labels or []) - set(model.labels))
    if missing:
        raise ValueError(
            f"Unknown DINO bank class label(s): {', '.join(missing)}. "
            f"Available labels: {', '.join(model.labels)}"
        )

    paths = _filter_class_labels(
        list_images(validation_image_dir),
        validation_image_dir,
        resolved_class_depth,
        class_labels,
    )
    if not paths:
        raise ValueError(f"No validation images found under {validation_image_dir}")

    embeddings, manifest, _ = _extract_raw_embeddings(
        paths=paths,
        output_dir=output_dir / "features",
        config=model.config,
        structural_scaler=model.structural_scaler,
        device=device,
    )
    vectors = _transform_vectors(embeddings, model)
    labels = infer_labels_from_root(paths, validation_image_dir, resolved_class_depth)
    ground_truth = [_ground_truth_label(path, validation_image_dir, resolved_class_depth) for path in paths]
    predictions = _score_known_labels(
        image_paths=paths,
        labels=labels,
        vectors=vectors,
        model=model,
        ground_truth=ground_truth,
    )
    predictions_path = output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    manifest["label"] = labels
    manifest["ground_truth"] = ground_truth
    manifest.to_csv(output_dir / "validation_manifest.csv", index=False)

    summary = _summarize_predictions(predictions)
    summary.to_csv(output_dir / "dinobank_validation_summary.csv", index=False)
    _plot_histogram(predictions, output_dir / "plots", model.config.histogram_bins)
    _export_montage(predictions, output_dir / "montage", model.config.montage_samples)
    _plot_umap(vectors, labels, predictions, output_dir / "plots", model.config.random_state)
    return _write_report(
        output_dir=output_dir,
        model_path=model_path,
        summary=summary,
        validation_report=None,
        predictions_path=predictions_path,
    )


def _extract_raw_embeddings(
    paths: list[Path],
    output_dir: Path,
    config: DinoBankConfig,
    structural_scaler: StandardScaler | None,
    device: str | None,
) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    embeddings, structural_summary, fitted_scaler = build_combined_embeddings(
        paths=paths,
        output_dir=output_dir,
        model_name=config.dinov2_model,
        image_size=config.image_size,
        batch_size=config.batch_size,
        structural_weight=config.structural_weight,
        projection_profile_dims=config.projection_profile_dims,
        bright_threshold=config.bright_threshold,
        edge_threshold=config.edge_threshold,
        peak_threshold_std=config.peak_threshold_std,
        peak_min_distance=config.peak_min_distance,
        structural_image_size=config.structural_image_size,
        structural_scaler=structural_scaler,
        device=device,
    )
    return embeddings, structural_summary, fitted_scaler


def _fit_pca(vectors: np.ndarray, config: DinoBankConfig) -> PCA | None:
    if vectors.shape[0] < 2 or config.pca_components <= 0:
        return None
    components = min(config.pca_components, vectors.shape[0] - 1, vectors.shape[1])
    if components < 1:
        return None
    return PCA(n_components=components, random_state=config.random_state)


def _transform_vectors(embeddings: np.ndarray, model: DinoBankModel) -> np.ndarray:
    scaled = model.feature_scaler.transform(embeddings)
    return model.pca.transform(scaled) if model.pca is not None else scaled


def _normal_training_paths(root: Path, class_depth: int) -> list[Path]:
    paths = []
    for path in list_images(root):
        if not _has_token(path, root, class_depth, ANOMALY_TOKENS):
            paths.append(path)
    return paths


def _filter_class_labels(
    paths: list[Path],
    root: Path,
    class_depth: int,
    class_labels: list[str] | None,
) -> list[Path]:
    if not class_labels:
        return paths
    wanted = set(class_labels)
    return [
        path
        for path in paths
        if infer_labels_from_root([path], root, class_depth)[0] in wanted
    ]


def _has_token(path: Path, root: Path, class_depth: int, tokens: set[str]) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    path_tokens = {part.lower() for part in relative.parts[class_depth:-1]}
    return bool(path_tokens & tokens)


def _ground_truth_label(path: Path, root: Path, class_depth: int) -> str:
    if _has_token(path, root, class_depth, ANOMALY_TOKENS):
        return "ng"
    if _has_token(path, root, class_depth, GOOD_TOKENS):
        return "good"
    return "good"


def _leave_one_out_distances(bank: np.ndarray) -> np.ndarray:
    if len(bank) <= 1:
        return np.zeros(len(bank), dtype=np.float32)
    distances = _pairwise_euclidean(bank, bank)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _score_known_labels(
    image_paths: list[Path],
    labels: list[str],
    vectors: np.ndarray,
    model: DinoBankModel,
    ground_truth: list[str] | None,
) -> pd.DataFrame:
    rows = []
    for index, (path, label) in enumerate(zip(image_paths, labels, strict=True)):
        if label not in model.banks:
            score = float("nan")
            threshold = float("nan")
            pred_label = "unknown_class"
        else:
            distances = _pairwise_euclidean(vectors[index : index + 1], model.banks[label])
            score = float(distances.min())
            threshold = float(model.thresholds[label])
            pred_label = "ng" if score > threshold else "good"
        row = {
            "image_path": str(path),
            "label": label,
            "pred_score": score,
            "threshold": threshold,
            "pred_label": pred_label,
        }
        if ground_truth is not None:
            row["ground_truth"] = ground_truth[index]
            row["is_correct"] = pred_label == ground_truth[index]
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in predictions.groupby("label"):
        row = {
            "label": label,
            "count": len(group),
            "score_mean": float(pd.to_numeric(group["pred_score"], errors="coerce").mean()),
            "score_p95": float(pd.to_numeric(group["pred_score"], errors="coerce").quantile(0.95)),
            "pred_ng_count": int((group["pred_label"] == "ng").sum()),
        }
        if "ground_truth" in group and set(group["ground_truth"].dropna()) <= {"good", "ng"}:
            y_true = (group["ground_truth"] == "ng").astype(int)
            y_pred = (group["pred_label"] == "ng").astype(int)
            row.update(_classification_metrics(y_true, y_pred, group["pred_score"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series, scores: pd.Series) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, pd.to_numeric(scores, errors="coerce")))
    return metrics


def _plot_histogram(predictions: pd.DataFrame, output_dir: Path, bins: int) -> None:
    scores = pd.to_numeric(predictions["pred_score"], errors="coerce")
    if scores.dropna().empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if "ground_truth" in predictions:
        for label, group in predictions.groupby("ground_truth"):
            pd.to_numeric(group["pred_score"], errors="coerce").dropna().hist(
                bins=bins,
                alpha=0.55,
                label=str(label),
            )
        plt.legend()
    else:
        scores.dropna().hist(bins=bins)
    plt.title("DINO bank anomaly scores")
    plt.xlabel("nearest-neighbor distance")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "score_histogram.png", dpi=160)
    plt.close()


def _export_montage(predictions: pd.DataFrame, output_dir: Path, samples: int) -> None:
    scored = predictions.copy()
    scored["pred_score"] = pd.to_numeric(scored["pred_score"], errors="coerce")
    scored = scored.dropna(subset=["pred_score"]).sort_values("pred_score", ascending=False)
    if scored.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = scored.head(samples)
    tile_size = (180, 120)
    columns = 5
    rows = max(1, (len(scored) + columns - 1) // columns)
    montage = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(montage)
    for index, row in enumerate(scored.itertuples(index=False)):
        image_path = Path(row.image_path)
        if not image_path.exists():
            continue
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            image.thumbnail(tile_size, Image.Resampling.BICUBIC)
            tile_x = (index % columns) * tile_size[0]
            tile_y = (index // columns) * tile_size[1]
            x = tile_x + (tile_size[0] - image.width) // 2
            y = tile_y + (tile_size[1] - image.height) // 2
            montage.paste(image, (x, y))
            draw.rectangle(
                [tile_x, tile_y, tile_x + tile_size[0] - 1, tile_y + tile_size[1] - 1],
                outline=(220, 0, 0),
                width=2,
            )
            draw.text((tile_x + 4, tile_y + 4), f"{row.pred_score:.3f}", fill=(0, 0, 0))
    montage.save(output_dir / "top_anomaly_scores.jpg")


def _plot_umap(
    vectors: np.ndarray,
    labels: list[str],
    predictions: pd.DataFrame,
    output_dir: Path,
    random_state: int,
) -> None:
    if len(vectors) < 3:
        return
    try:
        import umap
    except ImportError:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    n_neighbors = min(15, max(2, len(vectors) - 1))
    points = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=random_state,
    ).fit_transform(vectors)
    label_codes, label_names = pd.factorize(np.asarray(labels))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=label_codes,
        s=12,
        alpha=0.75,
        cmap="tab20",
    )
    colorbar = plt.colorbar(scatter)
    colorbar.set_ticks(range(len(label_names)))
    colorbar.set_ticklabels(label_names)
    if "pred_label" in predictions:
        ng_mask = predictions["pred_label"].to_numpy() == "ng"
        plt.scatter(
            points[ng_mask, 0],
            points[ng_mask, 1],
            facecolors="none",
            edgecolors="red",
            s=48,
            linewidths=1.2,
        )
    plt.title("DINO bank validation UMAP")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_umap.png", dpi=180)
    plt.close()


def _write_report(
    output_dir: Path,
    model_path: Path,
    summary: pd.DataFrame,
    validation_report: Path | None,
    predictions_path: Path | None = None,
) -> Path:
    lines = ["# DINOv2 + Structural Bank Report", ""]
    lines.append(f"- Model: `{model_path}`")
    if predictions_path is not None:
        lines.append(f"- Predictions: `{predictions_path}`")
    if validation_report is not None:
        lines.append(f"- Validation report: `{validation_report}`")
    lines.extend(["", "## Summary", ""])
    lines.extend(_dataframe_to_markdown(summary) if not summary.empty else ["No rows."])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `dinobank_model.joblib`: per-class image-level feature banks",
            "- `dinobank_summary.csv`: per-class thresholds and train distance stats",
            "- `predictions.csv`: validation scores when validation is run",
            "- `plots/score_histogram.png`: validation score distribution",
            "- `plots/validation_umap.png`: validation UMAP when enough samples exist",
            "- `montage/top_anomaly_scores.jpg`: highest-score validation samples",
        ]
    )
    report_path = output_dir / "dinobank_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _dataframe_to_markdown(df: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines
