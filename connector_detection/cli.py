from __future__ import annotations

from pathlib import Path

import typer

from connector_detection.centroid import assign_nearest_centroids, fit_nearest_centroids
from connector_detection.clustering import assign_existing_embeddings, fit_clusters
from connector_detection.config import load_config
from connector_detection.dual_branch import StructuralFusionConfig
from connector_detection.dual_branch import train_dual_branch as train_dual_branch_pipeline
from connector_detection.dual_branch import validate_dual_branch as validate_dual_branch_pipeline
from connector_detection.features import extract_embeddings
from connector_detection.patchcore import (
    AnomalibPatchcoreConfig,
    train_patchcore_per_class,
    validate_patchcore_per_class,
)
from connector_detection.review import export_review_samples
from connector_detection.structural_features import StructuralFeatureConfig
from connector_detection.voc import crop_pin_bands_from_voc
from connector_detection.visualize import plot_umap

app = typer.Typer(help="Connector anomaly research pipeline.")


@app.command()
def extract(config: Path, device: str | None = None) -> None:
    cfg = load_config(config)
    embedding_path, manifest_path = extract_embeddings(
        image_dir=cfg.image_dir,
        output_dir=cfg.output_dir,
        model_name=cfg.dinov2_model,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        structural_weight=cfg.structural_weight,
        projection_profile_dims=cfg.projection_profile_dims,
        bright_threshold=cfg.bright_threshold,
        edge_threshold=cfg.edge_threshold,
        peak_threshold_std=cfg.peak_threshold_std,
        peak_min_distance=cfg.peak_min_distance,
        device=device,
    )
    typer.echo(f"Saved {embedding_path}")
    typer.echo(f"Saved {manifest_path}")


@app.command()
def cluster(config: Path) -> None:
    cfg = load_config(config)
    clusters_path, model_path = fit_clusters(
        embeddings_path=cfg.output_dir / "embeddings.npy",
        manifest_path=cfg.output_dir / "manifest.csv",
        output_dir=cfg.output_dir,
        pca_components=cfg.pca_components,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        unknown_distance_quantile=cfg.unknown_distance_quantile,
        random_state=cfg.random_state,
    )
    typer.echo(f"Saved {clusters_path}")
    typer.echo(f"Saved {model_path}")


@app.command()
def review(config: Path) -> None:
    cfg = load_config(config)
    summary_path = export_review_samples(
        clusters_csv=cfg.output_dir / "clusters.csv",
        output_dir=cfg.output_dir / "review",
        samples_per_cluster=cfg.review_samples_per_cluster,
        random_state=cfg.random_state,
    )
    typer.echo(f"Saved {summary_path}")


@app.command()
def umap(config: Path) -> None:
    cfg = load_config(config)
    output_path = plot_umap(
        pca_embeddings_path=cfg.output_dir / "embeddings_pca.npy",
        clusters_csv=cfg.output_dir / "clusters.csv",
        output_path=cfg.output_dir / "umap_clusters.png",
        random_state=cfg.random_state,
    )
    typer.echo(f"Saved {output_path}")


@app.command()
def assign(
    embeddings: Path,
    model: Path,
    output: Path = Path("outputs/assignments.csv"),
) -> None:
    output_path = assign_existing_embeddings(embeddings, model, output)
    typer.echo(f"Saved {output_path}")


@app.command()
def crop_pin_bands(
    xml_dir: Path = typer.Option(..., help="Directory containing PASCAL VOC XML files."),
    image_dir: Path = typer.Option(..., help="Directory containing connector images."),
    output_dir: Path = typer.Option(
        Path("outputs/pin_band_crops"),
        help="Output directory. Crops are rotated to the same up orientation.",
    ),
    label: str | None = typer.Option(
        None,
        help="Only crop VOC objects with this label. By default, crop all objects.",
    ),
    padding: int = typer.Option(0, min=0, help="Extra pixels around each bounding box."),
    image_format: str = typer.Option("png", help="Crop image format, for example png or jpg."),
) -> None:
    manifest_path = crop_pin_bands_from_voc(
        xml_dir=xml_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        label=label,
        padding=padding,
        image_format=image_format,
    )
    typer.echo(f"Saved {manifest_path}")


@app.command()
def fit_centroids(
    config: Path,
    labeled_image_dir: Path | None = typer.Option(
        None,
        help="Root folder whose child folders are manual labels. Defaults to config image_dir.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Output directory. Defaults to config output_dir.",
    ),
    label_depth: int = typer.Option(
        1,
        min=1,
        help="How many parent folder levels form the manual label.",
    ),
    threshold_quantile: float = typer.Option(
        0.995,
        min=0.0,
        max=1.0,
        help="Per-label distance quantile used as unknown threshold.",
    ),
    device: str | None = None,
) -> None:
    cfg = load_config(config)
    feature_image_dir = labeled_image_dir or cfg.image_dir
    feature_output_dir = output_dir or cfg.output_dir
    embedding_path, manifest_path = extract_embeddings(
        image_dir=feature_image_dir,
        output_dir=feature_output_dir,
        model_name=cfg.dinov2_model,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        structural_weight=cfg.structural_weight,
        projection_profile_dims=cfg.projection_profile_dims,
        bright_threshold=cfg.bright_threshold,
        edge_threshold=cfg.edge_threshold,
        peak_threshold_std=cfg.peak_threshold_std,
        peak_min_distance=cfg.peak_min_distance,
        device=device,
    )
    assignments_path, summary_path, plot_path = fit_nearest_centroids(
        embeddings_path=embedding_path,
        manifest_path=manifest_path,
        image_dir=feature_image_dir,
        output_dir=feature_output_dir,
        label_depth=label_depth,
        threshold_quantile=threshold_quantile,
        random_state=cfg.random_state,
    )
    typer.echo(f"Saved {assignments_path}")
    typer.echo(f"Saved {summary_path}")
    typer.echo(f"Saved {plot_path}")


@app.command()
def assign_centroids(
    embeddings: Path,
    manifest: Path,
    model: Path,
    output: Path = Path("outputs/nearest_centroid_predictions.csv"),
) -> None:
    output_path = assign_nearest_centroids(
        embeddings_path=embeddings,
        manifest_path=manifest,
        model_path=model,
        output_path=output,
    )
    typer.echo(f"Saved {output_path}")


@app.command()
def train_patchcore(
    config: Path,
    train_image_dir: Path = typer.Option(
        ...,
        help="Root folder whose child folders are class labels. Use good images for each class.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/patchcore"),
        help="Output directory for models, validation CSVs, plots, and report.",
    ),
    validation_image_dir: Path | None = typer.Option(
        None,
        help="Optional validation root. Expected layout starts with the same class folders.",
    ),
    class_depth: int = typer.Option(
        1,
        min=1,
        help="How many path components under the root form the class label.",
    ),
    backbone: str | None = typer.Option(
        None,
        help="Override config patchcore_backbone.",
    ),
    layers: str | None = typer.Option(
        None,
        help="Comma-separated layers, for example layer2,layer3.",
    ),
    coreset_sampling_ratio: float | None = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Override config patchcore_coreset_sampling_ratio.",
    ),
    num_neighbors: int | None = typer.Option(
        None,
        min=1,
        help="Override config patchcore_num_neighbors.",
    ),
    train_batch_size: int | None = typer.Option(
        None,
        min=1,
        help="Override config patchcore_train_batch_size.",
    ),
    eval_batch_size: int | None = typer.Option(
        None,
        min=1,
        help="Override config patchcore_eval_batch_size.",
    ),
    num_workers: int | None = typer.Option(
        None,
        min=0,
        help="Override config patchcore_num_workers.",
    ),
    anomalib_image_size: int | None = typer.Option(
        None,
        min=1,
        help="Override config patchcore_image_size.",
    ),
    center_crop_size: int | None = typer.Option(
        None,
        min=1,
        help="Override config patchcore_center_crop_size.",
    ),
    histogram_bins: int | None = typer.Option(
        None,
        min=1,
        help="Override config patchcore_histogram_bins.",
    ),
    montage_samples: int | None = typer.Option(
        None,
        min=1,
        help="Override config patchcore_montage_samples.",
    ),
    device: str | None = None,
) -> None:
    cfg = load_config(config)
    anomalib_cfg = AnomalibPatchcoreConfig(
        backbone=backbone or cfg.patchcore_backbone,
        layers=tuple((layers.split(",") if layers else cfg.patchcore_layers)),
        coreset_sampling_ratio=(
            cfg.patchcore_coreset_sampling_ratio
            if coreset_sampling_ratio is None
            else coreset_sampling_ratio
        ),
        num_neighbors=cfg.patchcore_num_neighbors if num_neighbors is None else num_neighbors,
        train_batch_size=(
            cfg.patchcore_train_batch_size
            if train_batch_size is None
            else train_batch_size
        ),
        eval_batch_size=(
            cfg.patchcore_eval_batch_size if eval_batch_size is None else eval_batch_size
        ),
        num_workers=cfg.patchcore_num_workers if num_workers is None else num_workers,
        image_size=cfg.patchcore_image_size if anomalib_image_size is None else anomalib_image_size,
        center_crop_size=(
            cfg.patchcore_center_crop_size if center_crop_size is None else center_crop_size
        ),
        accelerator=cfg.patchcore_accelerator,
        devices=cfg.patchcore_devices,
        max_epochs=cfg.patchcore_max_epochs,
        normal_split_ratio=cfg.patchcore_normal_split_ratio,
        test_split_ratio=cfg.patchcore_test_split_ratio,
        val_split_ratio=cfg.patchcore_val_split_ratio,
        histogram_bins=(
            cfg.patchcore_histogram_bins if histogram_bins is None else histogram_bins
        ),
        montage_samples=(
            cfg.patchcore_montage_samples if montage_samples is None else montage_samples
        ),
        seed=cfg.random_state,
    )
    model_path, report_path = train_patchcore_per_class(
        train_image_dir=train_image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        validation_image_dir=validation_image_dir,
        config=anomalib_cfg,
    )
    typer.echo(f"Saved {model_path}")
    typer.echo(f"Saved {report_path}")


@app.command()
def validate_patchcore(
    config: Path,
    model: Path,
    validation_image_dir: Path = typer.Option(
        ...,
        help="Validation root. Expected layout starts with the same class folders.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/patchcore_validation"),
        help="Output directory for validation CSVs, plots, montage, and report.",
    ),
    class_depth: int = typer.Option(
        1,
        min=1,
        help="How many path components under the validation root form the class label.",
    ),
    device: str | None = None,
) -> None:
    cfg = load_config(config)
    anomalib_cfg = AnomalibPatchcoreConfig(
        backbone=cfg.patchcore_backbone,
        layers=cfg.patchcore_layers,
        coreset_sampling_ratio=cfg.patchcore_coreset_sampling_ratio,
        num_neighbors=cfg.patchcore_num_neighbors,
        train_batch_size=cfg.patchcore_train_batch_size,
        eval_batch_size=cfg.patchcore_eval_batch_size,
        num_workers=cfg.patchcore_num_workers,
        image_size=cfg.patchcore_image_size,
        center_crop_size=cfg.patchcore_center_crop_size,
        accelerator=cfg.patchcore_accelerator,
        devices=cfg.patchcore_devices,
        max_epochs=cfg.patchcore_max_epochs,
        normal_split_ratio=cfg.patchcore_normal_split_ratio,
        test_split_ratio=cfg.patchcore_test_split_ratio,
        val_split_ratio=cfg.patchcore_val_split_ratio,
        histogram_bins=cfg.patchcore_histogram_bins,
        montage_samples=cfg.patchcore_montage_samples,
        seed=cfg.random_state,
    )
    report_path = validate_patchcore_per_class(
        model_index_path=model,
        validation_image_dir=validation_image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        config=anomalib_cfg,
    )
    typer.echo(f"Saved {report_path}")


@app.command()
def train_dual_branch(
    config: Path,
    train_image_dir: Path = typer.Option(
        ...,
        help="Root folder whose child folders are class labels. Use good images for each class.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/dual_branch"),
        help="Output directory for PatchCore branch, structural branch, and fusion report.",
    ),
    validation_image_dir: Path | None = typer.Option(
        None,
        help="Optional validation root with the same class folder layout.",
    ),
    class_depth: int = typer.Option(
        1,
        min=1,
        help="How many path components under the root form the class label.",
    ),
) -> None:
    cfg = load_config(config)
    model_path, report_path = train_dual_branch_pipeline(
        train_image_dir=train_image_dir,
        output_dir=output_dir,
        patchcore_config=_patchcore_config_from_pipeline(cfg),
        structural_feature_config=_structural_config_from_pipeline(cfg),
        fusion_config=_fusion_config_from_pipeline(cfg),
        class_depth=class_depth,
        validation_image_dir=validation_image_dir,
    )
    typer.echo(f"Saved {model_path}")
    typer.echo(f"Saved {report_path}")


@app.command()
def validate_dual_branch(
    config: Path,
    model: Path,
    validation_image_dir: Path = typer.Option(
        ...,
        help="Validation root with the same class folder layout.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/dual_branch_validation"),
        help="Output directory for validation scores and report.",
    ),
    class_depth: int = typer.Option(
        1,
        min=1,
        help="How many path components under the validation root form the class label.",
    ),
) -> None:
    cfg = load_config(config)
    report_path = validate_dual_branch_pipeline(
        model_path=model,
        validation_image_dir=validation_image_dir,
        output_dir=output_dir,
        patchcore_config=_patchcore_config_from_pipeline(cfg),
        class_depth=class_depth,
    )
    typer.echo(f"Saved {report_path}")


def _patchcore_config_from_pipeline(cfg) -> AnomalibPatchcoreConfig:
    return AnomalibPatchcoreConfig(
        backbone=cfg.patchcore_backbone,
        layers=cfg.patchcore_layers,
        coreset_sampling_ratio=cfg.patchcore_coreset_sampling_ratio,
        num_neighbors=cfg.patchcore_num_neighbors,
        train_batch_size=cfg.patchcore_train_batch_size,
        eval_batch_size=cfg.patchcore_eval_batch_size,
        num_workers=cfg.patchcore_num_workers,
        image_size=cfg.patchcore_image_size,
        center_crop_size=cfg.patchcore_center_crop_size,
        accelerator=cfg.patchcore_accelerator,
        devices=cfg.patchcore_devices,
        max_epochs=cfg.patchcore_max_epochs,
        normal_split_ratio=cfg.patchcore_normal_split_ratio,
        test_split_ratio=cfg.patchcore_test_split_ratio,
        val_split_ratio=cfg.patchcore_val_split_ratio,
        histogram_bins=cfg.patchcore_histogram_bins,
        montage_samples=cfg.patchcore_montage_samples,
        seed=cfg.random_state,
    )


def _structural_config_from_pipeline(cfg) -> StructuralFeatureConfig:
    return StructuralFeatureConfig(
        projection_dims=cfg.projection_profile_dims,
        bright_threshold=cfg.bright_threshold,
        edge_threshold=cfg.edge_threshold,
        peak_threshold_std=cfg.peak_threshold_std,
        peak_min_distance=cfg.peak_min_distance,
    )


def _fusion_config_from_pipeline(cfg) -> StructuralFusionConfig:
    return StructuralFusionConfig(
        threshold_quantile=cfg.structural_score_threshold_quantile,
        peak_count_weight=cfg.fusion_peak_count_weight,
        peak_spacing_weight=cfg.fusion_peak_spacing_weight,
        profile_weight=cfg.fusion_profile_weight,
        metal_ratio_weight=cfg.fusion_metal_ratio_weight,
        patchcore_weight=cfg.fusion_patchcore_weight,
        fusion_threshold=cfg.fusion_threshold,
    )


@app.command()
def run(config: Path, device: str | None = None) -> None:
    extract(config, device)
    cluster(config)
    review(config)
    umap(config)
