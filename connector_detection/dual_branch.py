from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from connector_detection.images import list_images
from connector_detection.patchcore import AnomalibPatchcoreConfig, train_patchcore_per_class
from connector_detection.patchcore import infer_labels_from_root, validate_patchcore_per_class
from connector_detection.structural_features import (
    BASE_FEATURE_NAMES,
    StructuralFeatureConfig,
    compute_structural_features,
)


@dataclass(frozen=True)
class StructuralFusionConfig:
    threshold_quantile: float = 0.995
    peak_count_weight: float = 1.0
    peak_spacing_weight: float = 1.0
    profile_weight: float = 1.0
    metal_ratio_weight: float = 1.0
    patchcore_weight: float = 1.0
    fusion_threshold: float = 1.0


@dataclass
class StructuralProfile:
    label: str
    count: int
    scalar_mean: dict[str, float]
    scalar_std: dict[str, float]
    profile_mean: np.ndarray
    profile_std: np.ndarray
    structural_threshold: float


@dataclass
class DualBranchModel:
    patchcore_model_path: str
    structural_profiles: dict[str, StructuralProfile]
    structural_feature_config: StructuralFeatureConfig
    fusion_config: StructuralFusionConfig
    class_depth: int


def train_dual_branch(
    train_image_dir: Path,
    output_dir: Path,
    patchcore_config: AnomalibPatchcoreConfig,
    structural_feature_config: StructuralFeatureConfig,
    fusion_config: StructuralFusionConfig,
    class_depth: int = 1,
    validation_image_dir: Path | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    structural_dir = output_dir / "structural_branch"
    patchcore_dir = output_dir / "patchcore_branch"

    profiles, train_scores = fit_structural_profiles(
        image_dir=train_image_dir,
        output_dir=structural_dir / "train",
        feature_config=structural_feature_config,
        fusion_config=fusion_config,
        class_depth=class_depth,
    )
    profile_path = structural_dir / "structural_profiles.joblib"
    joblib.dump(profiles, profile_path)

    patchcore_model_path, patchcore_report_path = train_patchcore_per_class(
        train_image_dir=train_image_dir,
        output_dir=patchcore_dir,
        class_depth=class_depth,
        validation_image_dir=validation_image_dir,
        config=patchcore_config,
    )

    validation_scores = None
    fusion_scores = None
    if validation_image_dir is not None:
        validation_scores = score_structural_branch(
            image_dir=validation_image_dir,
            output_dir=structural_dir / "validation",
            profiles=profiles,
            feature_config=structural_feature_config,
            fusion_config=fusion_config,
            class_depth=class_depth,
        )
        fusion_scores = fuse_dual_branch_scores(
            structural_scores=validation_scores,
            patchcore_output_dir=patchcore_dir,
            output_path=output_dir / "dual_branch_fusion_scores.csv",
            fusion_config=fusion_config,
        )

    model = DualBranchModel(
        patchcore_model_path=str(patchcore_model_path),
        structural_profiles=profiles,
        structural_feature_config=structural_feature_config,
        fusion_config=fusion_config,
        class_depth=class_depth,
    )
    model_path = output_dir / "dual_branch_model.joblib"
    joblib.dump(model, model_path)

    report_path = write_dual_branch_report(
        output_dir=output_dir,
        profiles=profiles,
        train_scores=train_scores,
        validation_scores=validation_scores,
        fusion_scores=fusion_scores,
        patchcore_report_path=patchcore_report_path,
        model_path=model_path,
        profile_path=profile_path,
    )
    return model_path, report_path


def validate_dual_branch(
    model_path: Path,
    validation_image_dir: Path,
    output_dir: Path,
    patchcore_config: AnomalibPatchcoreConfig,
    class_depth: int | None = None,
) -> Path:
    model: DualBranchModel = joblib.load(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    structural_scores = score_structural_branch(
        image_dir=validation_image_dir,
        output_dir=output_dir / "structural_branch",
        profiles=model.structural_profiles,
        feature_config=model.structural_feature_config,
        fusion_config=model.fusion_config,
        class_depth=model.class_depth if class_depth is None else class_depth,
    )
    patchcore_report = validate_patchcore_per_class(
        model_index_path=Path(model.patchcore_model_path),
        validation_image_dir=validation_image_dir,
        output_dir=output_dir / "patchcore_branch",
        class_depth=model.class_depth if class_depth is None else class_depth,
        config=patchcore_config,
    )
    fusion_scores = fuse_dual_branch_scores(
        structural_scores=structural_scores,
        patchcore_output_dir=output_dir / "patchcore_branch",
        output_path=output_dir / "dual_branch_fusion_scores.csv",
        fusion_config=model.fusion_config,
    )
    return write_dual_branch_report(
        output_dir=output_dir,
        profiles=model.structural_profiles,
        train_scores=None,
        validation_scores=structural_scores,
        fusion_scores=fusion_scores,
        patchcore_report_path=patchcore_report,
        model_path=model_path,
        profile_path=None,
    )


def fit_structural_profiles(
    image_dir: Path,
    output_dir: Path,
    feature_config: StructuralFeatureConfig,
    fusion_config: StructuralFusionConfig,
    class_depth: int,
) -> tuple[dict[str, StructuralProfile], pd.DataFrame]:
    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")
    features, summary = compute_structural_features(image_paths, feature_config)
    labels = infer_labels_from_root(image_paths, image_dir, class_depth)
    summary["label"] = labels
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "structural_features.npy", features)
    summary.to_csv(output_dir / "structural_summary.csv", index=False)

    projection = features[:, len(BASE_FEATURE_NAMES) :]
    profiles: dict[str, StructuralProfile] = {}
    train_score_parts = []
    for label in sorted(set(labels)):
        mask = np.asarray(labels) == label
        group = summary[mask].reset_index(drop=True)
        profile_mean = projection[mask].mean(axis=0)
        profile_std = projection[mask].std(axis=0)
        scalar_mean = {column: float(group[column].mean()) for column in BASE_FEATURE_NAMES}
        scalar_std = {
            column: float(max(group[column].std(ddof=0), 1e-6))
            for column in BASE_FEATURE_NAMES
        }
        provisional = StructuralProfile(
            label=label,
            count=int(mask.sum()),
            scalar_mean=scalar_mean,
            scalar_std=scalar_std,
            profile_mean=profile_mean,
            profile_std=profile_std,
            structural_threshold=float("inf"),
        )
        scores = _score_structural_rows(
            summary=group,
            projection=projection[mask],
            profile=provisional,
            fusion_config=fusion_config,
        )
        threshold = float(np.quantile(scores["structural_score"], fusion_config.threshold_quantile))
        profiles[label] = StructuralProfile(
            **{**provisional.__dict__, "structural_threshold": threshold}
        )
        train_score_parts.append(scores)

    train_scores = pd.concat(train_score_parts, ignore_index=True)
    train_scores["predicted_structural_ng"] = (
        train_scores["structural_score"]
        > train_scores["label"].map({label: p.structural_threshold for label, p in profiles.items()})
    )
    train_scores.to_csv(output_dir / "structural_train_scores.csv", index=False)
    _write_profile_summary(profiles, output_dir / "structural_profile_summary.csv")
    return profiles, train_scores


def score_structural_branch(
    image_dir: Path,
    output_dir: Path,
    profiles: dict[str, StructuralProfile],
    feature_config: StructuralFeatureConfig,
    fusion_config: StructuralFusionConfig,
    class_depth: int,
) -> pd.DataFrame:
    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")
    features, summary = compute_structural_features(image_paths, feature_config)
    labels = infer_labels_from_root(image_paths, image_dir, class_depth)
    summary["label"] = labels
    projection = features[:, len(BASE_FEATURE_NAMES) :]
    parts = []
    for label in sorted(set(labels)):
        if label not in profiles:
            continue
        mask = np.asarray(labels) == label
        parts.append(
            _score_structural_rows(
                summary=summary[mask].reset_index(drop=True),
                projection=projection[mask],
                profile=profiles[label],
                fusion_config=fusion_config,
            )
        )
    scores = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    thresholds = {label: profile.structural_threshold for label, profile in profiles.items()}
    scores["structural_threshold"] = scores["label"].map(thresholds)
    scores["predicted_structural_ng"] = scores["structural_score"] > scores["structural_threshold"]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "structural_features.npy", features)
    summary.to_csv(output_dir / "structural_summary.csv", index=False)
    scores.to_csv(output_dir / "structural_scores.csv", index=False)
    return scores


def _score_structural_rows(
    summary: pd.DataFrame,
    projection: np.ndarray,
    profile: StructuralProfile,
    fusion_config: StructuralFusionConfig,
) -> pd.DataFrame:
    rows = summary.copy()
    rows["peak_count_deviation"] = _z_abs(
        rows["peak_count"],
        profile.scalar_mean["peak_count"],
        profile.scalar_std["peak_count"],
    )
    spacing_mean = _z_abs(
        rows["peak_spacing_mean"],
        profile.scalar_mean["peak_spacing_mean"],
        profile.scalar_std["peak_spacing_mean"],
    )
    spacing_std = _z_abs(
        rows["peak_spacing_std"],
        profile.scalar_mean["peak_spacing_std"],
        profile.scalar_std["peak_spacing_std"],
    )
    rows["peak_spacing_anomaly"] = 0.5 * spacing_mean + 0.5 * spacing_std
    rows["profile_similarity_anomaly"] = [
        _profile_distance(vector, profile.profile_mean) for vector in projection
    ]
    rows["metal_ratio_anomaly"] = _z_abs(
        rows["bright_ratio"],
        profile.scalar_mean["bright_ratio"],
        profile.scalar_std["bright_ratio"],
    )
    rows["structural_score"] = (
        fusion_config.peak_count_weight * rows["peak_count_deviation"]
        + fusion_config.peak_spacing_weight * rows["peak_spacing_anomaly"]
        + fusion_config.profile_weight * rows["profile_similarity_anomaly"]
        + fusion_config.metal_ratio_weight * rows["metal_ratio_anomaly"]
    )
    return rows


def fuse_dual_branch_scores(
    structural_scores: pd.DataFrame,
    patchcore_output_dir: Path,
    output_path: Path,
    fusion_config: StructuralFusionConfig,
) -> pd.DataFrame:
    patchcore_scores = _read_patchcore_prediction_scores(patchcore_output_dir)
    fused = structural_scores.copy()
    if not patchcore_scores.empty:
        fused = fused.merge(patchcore_scores, on="image_path", how="left")
    if "pred_score" not in fused:
        fused["pred_score"] = np.nan
    normalized_patchcore = _normalize_series(pd.to_numeric(fused["pred_score"], errors="coerce"))
    normalized_structural = _normalize_series(fused["structural_score"])
    fused["fusion_score"] = (
        fusion_config.patchcore_weight * normalized_patchcore
        + normalized_structural
    )
    fused["predicted_ng"] = (
        fused["predicted_structural_ng"]
        | (fused["fusion_score"] > fusion_config.fusion_threshold)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(output_path, index=False)
    return fused


def _read_patchcore_prediction_scores(patchcore_output_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in patchcore_output_dir.rglob("predictions.csv"):
        df = pd.read_csv(csv_path)
        if "image_path" in df.columns:
            for column in ("pred_score", "raw_heatmap_path", "heatmap_path", "overlay_path"):
                if column not in df.columns:
                    df[column] = np.nan if column == "pred_score" else ""
            rows.append(
                df[
                    [
                        "image_path",
                        "pred_score",
                        "raw_heatmap_path",
                        "heatmap_path",
                        "overlay_path",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _z_abs(values: pd.Series, mean: float, std: float) -> pd.Series:
    return (values.astype(float) - mean).abs() / max(std, 1e-6)


def _profile_distance(vector: np.ndarray, mean_profile: np.ndarray) -> float:
    numerator = float(np.dot(vector, mean_profile))
    denominator = float(np.linalg.norm(vector) * np.linalg.norm(mean_profile))
    if denominator <= 1e-12:
        return 0.0
    return max(0.0, 1.0 - numerator / denominator)


def _normalize_series(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    if values.dropna().empty:
        return pd.Series(np.zeros(len(values)), index=values.index)
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return values.fillna(0.0) * 0.0
    return ((values - lo) / (hi - lo)).fillna(0.0)


def _write_profile_summary(profiles: dict[str, StructuralProfile], path: Path) -> None:
    rows = []
    for label, profile in profiles.items():
        rows.append(
            {
                "label": label,
                "count": profile.count,
                "peak_count_mean": profile.scalar_mean["peak_count"],
                "peak_count_std": profile.scalar_std["peak_count"],
                "peak_spacing_mean": profile.scalar_mean["peak_spacing_mean"],
                "bright_ratio_mean": profile.scalar_mean["bright_ratio"],
                "structural_threshold": profile.structural_threshold,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_dual_branch_report(
    output_dir: Path,
    profiles: dict[str, StructuralProfile],
    train_scores: pd.DataFrame | None,
    validation_scores: pd.DataFrame | None,
    fusion_scores: pd.DataFrame | None,
    patchcore_report_path: Path,
    model_path: Path,
    profile_path: Path | None,
) -> Path:
    lines = ["# Dual Branch Report", ""]
    lines.append("Architecture: anomalib PatchCore branch + structural profile branch")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Dual model: `{model_path}`")
    if profile_path is not None:
        lines.append(f"- Structural profiles: `{profile_path}`")
    lines.append(f"- PatchCore report: `{patchcore_report_path}`")
    lines.append("- Structural scores: `structural_branch/**/structural_scores.csv`")
    lines.append("- Fusion scores: `dual_branch_fusion_scores.csv`")
    lines.append("")
    lines.append("## Structural Profiles")
    lines.append("")
    lines.append("| label | count | peak_count_mean | bright_ratio_mean | threshold |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label, profile in profiles.items():
        lines.append(
            f"| {label} | {profile.count} | {profile.scalar_mean['peak_count']:.3f} | "
            f"{profile.scalar_mean['bright_ratio']:.6f} | {profile.structural_threshold:.6f} |"
        )
    if train_scores is not None:
        lines.append("")
        lines.append(f"- Train images scored by structural branch: {len(train_scores)}")
    if validation_scores is not None:
        lines.append(f"- Validation images scored by structural branch: {len(validation_scores)}")
    if fusion_scores is not None:
        lines.append(f"- Fusion predicted NG: {int(fusion_scores['predicted_ng'].sum())}")
    report_path = output_dir / "dual_branch_report.md"
    report_path.write_text("\n".join(lines))
    return report_path
