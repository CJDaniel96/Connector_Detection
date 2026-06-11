from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


ImageSize = int | tuple[int, int]


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
    dinobank_pca_components: int = 50
    dinobank_threshold_quantile: float = 0.995
    dinobank_histogram_bins: int = 30
    dinobank_montage_samples: int = 30
    dinomaly_encoder_name: str = "dinov2reg_vit_base_14"
    dinomaly_bottleneck_dropout: float = 0.2
    dinomaly_decoder_depth: int = 8
    dinomaly_target_layers: tuple[int, ...] | None = None
    dinomaly_fuse_layer_encoder: tuple[tuple[int, ...], ...] | None = None
    dinomaly_fuse_layer_decoder: tuple[tuple[int, ...], ...] | None = None
    dinomaly_remove_class_token: bool = False
    dinomaly_use_context_recentering: bool = False
    dinomaly_train_batch_size: int = 16
    dinomaly_eval_batch_size: int = 16
    dinomaly_num_workers: int = 0
    dinomaly_image_size: ImageSize | None = (448, 448)
    dinomaly_crop_size: int | None = 392
    dinomaly_accelerator: str = "auto"
    dinomaly_devices: str = "auto"
    dinomaly_max_epochs: int = 10
    dinomaly_normal_split_ratio: float = 0.2
    dinomaly_test_split_ratio: float = 0.2
    dinomaly_val_split_ratio: float = 0.5
    dinomaly_histogram_bins: int = 30
    dinomaly_montage_samples: int = 30
    dinomaly_heatmap_alpha: float = 0.45
    projection_profile_dims: int = 128
    bright_threshold: float = 0.65
    edge_threshold: float = 0.12
    peak_threshold_std: float = 0.5
    peak_min_distance: int = 3
    structural_image_size: ImageSize | None = None
    patchcore_backbone: str = "wide_resnet50_2"
    patchcore_layers: tuple[str, ...] = ("layer2", "layer3")
    patchcore_coreset_sampling_ratio: float = 0.1
    patchcore_num_neighbors: int = 9
    patchcore_train_batch_size: int = 32
    patchcore_eval_batch_size: int = 32
    patchcore_num_workers: int = 0
    patchcore_image_size: ImageSize = 256
    patchcore_center_crop_size: ImageSize | None = 224
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
        dinobank_pca_components=int(section.get("dinobank_pca_components", section.get("pca_components", 50))),
        dinobank_threshold_quantile=float(section.get("dinobank_threshold_quantile", 0.995)),
        dinobank_histogram_bins=int(section.get("dinobank_histogram_bins", 30)),
        dinobank_montage_samples=int(section.get("dinobank_montage_samples", 30)),
        dinomaly_encoder_name=str(section.get("dinomaly_encoder_name", "dinov2reg_vit_base_14")),
        dinomaly_bottleneck_dropout=float(section.get("dinomaly_bottleneck_dropout", 0.2)),
        dinomaly_decoder_depth=int(section.get("dinomaly_decoder_depth", 8)),
        dinomaly_target_layers=_parse_optional_int_tuple(section.get("dinomaly_target_layers")),
        dinomaly_fuse_layer_encoder=_parse_optional_nested_int_tuple(
            section.get("dinomaly_fuse_layer_encoder")
        ),
        dinomaly_fuse_layer_decoder=_parse_optional_nested_int_tuple(
            section.get("dinomaly_fuse_layer_decoder")
        ),
        dinomaly_remove_class_token=bool(section.get("dinomaly_remove_class_token", False)),
        dinomaly_use_context_recentering=bool(
            section.get("dinomaly_use_context_recentering", False)
        ),
        dinomaly_train_batch_size=int(section.get("dinomaly_train_batch_size", 16)),
        dinomaly_eval_batch_size=int(section.get("dinomaly_eval_batch_size", 16)),
        dinomaly_num_workers=int(section.get("dinomaly_num_workers", 0)),
        dinomaly_image_size=_parse_optional_image_size(
            section.get("dinomaly_image_size", [448, 448])
        ),
        dinomaly_crop_size=(
            None
            if section.get("dinomaly_crop_size", 392) in (None, "null")
            else int(section.get("dinomaly_crop_size", 392))
        ),
        dinomaly_accelerator=str(section.get("dinomaly_accelerator", "auto")),
        dinomaly_devices=str(section.get("dinomaly_devices", "auto")),
        dinomaly_max_epochs=int(section.get("dinomaly_max_epochs", 10)),
        dinomaly_normal_split_ratio=float(section.get("dinomaly_normal_split_ratio", 0.2)),
        dinomaly_test_split_ratio=float(section.get("dinomaly_test_split_ratio", 0.2)),
        dinomaly_val_split_ratio=float(section.get("dinomaly_val_split_ratio", 0.5)),
        dinomaly_histogram_bins=int(section.get("dinomaly_histogram_bins", 30)),
        dinomaly_montage_samples=int(section.get("dinomaly_montage_samples", 30)),
        dinomaly_heatmap_alpha=float(section.get("dinomaly_heatmap_alpha", 0.45)),
        projection_profile_dims=int(section.get("projection_profile_dims", 128)),
        bright_threshold=float(section.get("bright_threshold", 0.65)),
        edge_threshold=float(section.get("edge_threshold", 0.12)),
        peak_threshold_std=float(section.get("peak_threshold_std", 0.5)),
        peak_min_distance=int(section.get("peak_min_distance", 3)),
        structural_image_size=_parse_optional_image_size(section.get("structural_image_size")),
        patchcore_backbone=str(section.get("patchcore_backbone", "wide_resnet50_2")),
        patchcore_layers=tuple(section.get("patchcore_layers", ["layer2", "layer3"])),
        patchcore_coreset_sampling_ratio=float(
            section.get("patchcore_coreset_sampling_ratio", 0.1)
        ),
        patchcore_num_neighbors=int(section.get("patchcore_num_neighbors", 9)),
        patchcore_train_batch_size=int(section.get("patchcore_train_batch_size", 32)),
        patchcore_eval_batch_size=int(section.get("patchcore_eval_batch_size", 32)),
        patchcore_num_workers=int(section.get("patchcore_num_workers", 0)),
        patchcore_image_size=_parse_image_size(section.get("patchcore_image_size", 256)),
        patchcore_center_crop_size=_parse_optional_image_size(
            section.get("patchcore_center_crop_size", 224)
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


def _parse_image_size(value: object) -> ImageSize:
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("Image size list must be [height, width].")
        return int(value[0]), int(value[1])
    return int(value)


def _parse_optional_image_size(value: object) -> ImageSize | None:
    if value in (None, "null"):
        return None
    return _parse_image_size(value)


def _parse_optional_int_tuple(value: object) -> tuple[int, ...] | None:
    if value in (None, "null"):
        return None
    if not isinstance(value, list):
        raise ValueError("Integer tuple config values must be TOML lists.")
    return tuple(int(item) for item in value)


def _parse_optional_nested_int_tuple(value: object) -> tuple[tuple[int, ...], ...] | None:
    if value in (None, "null"):
        return None
    if not isinstance(value, list):
        raise ValueError("Nested integer tuple config values must be TOML lists.")
    return tuple(tuple(int(item) for item in group) for group in value)
