from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PipelineConfig:
    image_dir: Path
    output_dir: Path
    image_size: int = 224
    dinov2_model: str = "facebook/dinov2-small"
    batch_size: int = 16
    pca_components: int = 50
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int | None = None
    review_samples_per_cluster: int = 30
    unknown_distance_quantile: float = 0.995
    structural_weight: float = 2.0
    projection_profile_dims: int = 128
    bright_threshold: float = 0.65
    edge_threshold: float = 0.12
    peak_threshold_std: float = 0.5
    peak_min_distance: int = 3
    patchcore_backbone: str = "wide_resnet50_2"
    patchcore_layers: tuple[str, ...] = ("layer2", "layer3")
    patchcore_coreset_sampling_ratio: float = 0.1
    patchcore_num_neighbors: int = 9
    patchcore_train_batch_size: int = 32
    patchcore_eval_batch_size: int = 32
    patchcore_num_workers: int = 0
    patchcore_image_size: int = 256
    patchcore_center_crop_size: int | None = 224
    patchcore_accelerator: str = "auto"
    patchcore_devices: str = "auto"
    patchcore_max_epochs: int = 1
    patchcore_normal_split_ratio: float = 0.2
    patchcore_test_split_ratio: float = 0.2
    patchcore_val_split_ratio: float = 0.5
    patchcore_histogram_bins: int = 30
    patchcore_montage_samples: int = 30
    structural_score_threshold_quantile: float = 0.995
    fusion_patchcore_weight: float = 1.0
    fusion_peak_count_weight: float = 1.0
    fusion_peak_spacing_weight: float = 1.0
    fusion_profile_weight: float = 1.0
    fusion_metal_ratio_weight: float = 1.0
    fusion_threshold: float = 1.0
    random_state: int = 42


def load_config(path: Path) -> PipelineConfig:
    data = tomllib.loads(path.read_text())
    section = data.get("pipeline", data)
    return PipelineConfig(
        image_dir=Path(section["image_dir"]).expanduser(),
        output_dir=Path(section.get("output_dir", "outputs")).expanduser(),
        image_size=int(section.get("image_size", 224)),
        dinov2_model=str(section.get("dinov2_model", "facebook/dinov2-small")),
        batch_size=int(section.get("batch_size", 16)),
        pca_components=int(section.get("pca_components", 50)),
        hdbscan_min_cluster_size=int(section.get("hdbscan_min_cluster_size", 15)),
        hdbscan_min_samples=(
            None
            if section.get("hdbscan_min_samples") in (None, "null")
            else int(section["hdbscan_min_samples"])
        ),
        review_samples_per_cluster=int(section.get("review_samples_per_cluster", 30)),
        unknown_distance_quantile=float(section.get("unknown_distance_quantile", 0.995)),
        structural_weight=float(section.get("structural_weight", 2.0)),
        projection_profile_dims=int(section.get("projection_profile_dims", 128)),
        bright_threshold=float(section.get("bright_threshold", 0.65)),
        edge_threshold=float(section.get("edge_threshold", 0.12)),
        peak_threshold_std=float(section.get("peak_threshold_std", 0.5)),
        peak_min_distance=int(section.get("peak_min_distance", 3)),
        patchcore_backbone=str(section.get("patchcore_backbone", "wide_resnet50_2")),
        patchcore_layers=tuple(section.get("patchcore_layers", ["layer2", "layer3"])),
        patchcore_coreset_sampling_ratio=float(
            section.get("patchcore_coreset_sampling_ratio", 0.1)
        ),
        patchcore_num_neighbors=int(section.get("patchcore_num_neighbors", 9)),
        patchcore_train_batch_size=int(section.get("patchcore_train_batch_size", 32)),
        patchcore_eval_batch_size=int(section.get("patchcore_eval_batch_size", 32)),
        patchcore_num_workers=int(section.get("patchcore_num_workers", 0)),
        patchcore_image_size=int(section.get("patchcore_image_size", 256)),
        patchcore_center_crop_size=(
            None
            if section.get("patchcore_center_crop_size") in (None, "null")
            else int(section.get("patchcore_center_crop_size", 224))
        ),
        patchcore_accelerator=str(section.get("patchcore_accelerator", "auto")),
        patchcore_devices=str(section.get("patchcore_devices", "auto")),
        patchcore_max_epochs=int(section.get("patchcore_max_epochs", 1)),
        patchcore_normal_split_ratio=float(section.get("patchcore_normal_split_ratio", 0.2)),
        patchcore_test_split_ratio=float(section.get("patchcore_test_split_ratio", 0.2)),
        patchcore_val_split_ratio=float(section.get("patchcore_val_split_ratio", 0.5)),
        patchcore_histogram_bins=int(section.get("patchcore_histogram_bins", 30)),
        patchcore_montage_samples=int(section.get("patchcore_montage_samples", 30)),
        structural_score_threshold_quantile=float(
            section.get("structural_score_threshold_quantile", 0.995)
        ),
        fusion_patchcore_weight=float(section.get("fusion_patchcore_weight", 1.0)),
        fusion_peak_count_weight=float(section.get("fusion_peak_count_weight", 1.0)),
        fusion_peak_spacing_weight=float(section.get("fusion_peak_spacing_weight", 1.0)),
        fusion_profile_weight=float(section.get("fusion_profile_weight", 1.0)),
        fusion_metal_ratio_weight=float(section.get("fusion_metal_ratio_weight", 1.0)),
        fusion_threshold=float(section.get("fusion_threshold", 1.0)),
        random_state=int(section.get("random_state", 42)),
    )
