from __future__ import annotations

from pathlib import Path

import typer

from connector_detection.centroid import assign_nearest_centroids, fit_nearest_centroids
from connector_detection.clustering import assign_existing_embeddings, fit_clusters
from connector_detection.compare import compare_baselines as compare_baselines_pipeline
from connector_detection.config import load_config
from connector_detection.dual_branch import StructuralFusionConfig
from connector_detection.dual_branch import train_dual_branch as train_dual_branch_pipeline
from connector_detection.dual_branch import validate_dual_branch as validate_dual_branch_pipeline
from connector_detection.dinomaly import (
    AnomalibDinomalyConfig,
    predict_dinomaly_blind,
    train_dinomaly_unified,
    validate_dinomaly_unified,
)
from connector_detection.dinobank import DinoBankConfig, train_dinobank, validate_dinobank
from connector_detection.features import extract_embeddings
from connector_detection.patchcore import (
    AnomalibPatchcoreConfig,
    train_patchcore_per_class,
    validate_patchcore_per_class,
)
from connector_detection.review import export_review_samples
from connector_detection.structural_features import StructuralFeatureConfig
from connector_detection.voc import crop_pin_bands_from_voc, rotate_images_from_voc_orientation
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
        structural_image_size=cfg.structural_image_size,
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
def rotate_images_by_voc(
    xml_dir: Path = typer.Option(..., help="Directory containing PASCAL VOC XML files."),
    image_dir: Path = typer.Option(..., help="Directory containing source images."),
    output_dir: Path = typer.Option(
        Path("outputs/rotated_images"),
        help="Output directory for full images rotated to the target orientation.",
    ),
    label: str = typer.Option(
        ...,
        help="VOC class name whose bounding box position defines the source orientation.",
    ),
    target_orientation: str = typer.Option(
        "up",
        help="Target orientation: up, right, down, or left.",
    ),
    image_format: str = typer.Option("png", help="Output image format, for example png or jpg."),
) -> None:
    manifest_path = rotate_images_from_voc_orientation(
        xml_dir=xml_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        label=label,
        target_orientation=target_orientation,
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
        structural_image_size=cfg.structural_image_size,
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


@app.command("patchcore-train")
@app.command("train-patchcore", hidden=True)
def patchcore_train(
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
    device: str | None = None,
) -> None:
    cfg = load_config(config)
    anomalib_cfg = _patchcore_config_from_pipeline(
        cfg,
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        image_size=anomalib_image_size,
        center_crop_size=center_crop_size,
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


@app.command("patchcore-validate")
@app.command("validate-patchcore", hidden=True)
def patchcore_validate(
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
    class_depth: int | None = typer.Option(
        None,
        min=1,
        help="Override the class_depth saved in the trained model index.",
    ),
    class_label: list[str] | None = typer.Option(
        None,
        "--class-label",
        help="Validate only this class label. Repeat the option to validate multiple classes.",
    ),
    device: str | None = None,
) -> None:
    load_config(config)
    report_path = validate_patchcore_per_class(
        model_index_path=model,
        validation_image_dir=validation_image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        class_labels=class_label,
    )
    typer.echo(f"Saved {report_path}")


@app.command("dinobank-train")
def dinobank_train(
    config: Path,
    train_image_dir: Path = typer.Option(
        ...,
        help="Root folder whose child folders are class labels. Normal images build the bank.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/dinobank"),
        help="Output directory for DINOv2 + structural bank artifacts.",
    ),
    validation_image_dir: Path | None = typer.Option(
        None,
        help="Optional validation root with matching class folders.",
    ),
    class_depth: int = typer.Option(
        1,
        min=1,
        help="How many path components under the root form the class label.",
    ),
    class_label: list[str] | None = typer.Option(
        None,
        "--class-label",
        help="Train only this class label. Repeat the option to train multiple classes.",
    ),
    device: str | None = None,
) -> None:
    cfg = load_config(config)
    model_path, report_path = train_dinobank(
        train_image_dir=train_image_dir,
        output_dir=output_dir,
        config=_dinobank_config_from_pipeline(cfg),
        class_depth=class_depth,
        validation_image_dir=validation_image_dir,
        class_labels=class_label,
        device=device,
    )
    typer.echo(f"Saved {model_path}")
    typer.echo(f"Saved {report_path}")


@app.command("dinobank-validate")
def dinobank_validate(
    config: Path,
    model: Path,
    validation_image_dir: Path = typer.Option(
        ...,
        help="Validation root with matching class folders.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/dinobank_validation"),
        help="Output directory for DINO bank validation artifacts.",
    ),
    class_depth: int | None = typer.Option(
        None,
        min=1,
        help="Override the class_depth saved in the trained model.",
    ),
    class_label: list[str] | None = typer.Option(
        None,
        "--class-label",
        help="Validate only this class label. Repeat the option to validate multiple classes.",
    ),
    device: str | None = None,
) -> None:
    load_config(config)
    report_path = validate_dinobank(
        model_path=model,
        validation_image_dir=validation_image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        class_labels=class_label,
        device=device,
    )
    typer.echo(f"Saved {report_path}")


@app.command("compare-baselines")
def compare_baselines(
    patchcore_predictions: Path = typer.Option(
        ...,
        help="PatchCore predictions.csv or output directory containing predictions.csv files.",
    ),
    dinobank_predictions: Path = typer.Option(
        ...,
        help="DINO bank predictions.csv or output directory containing predictions.csv files.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/baseline_comparison"),
        help="Output directory for comparison CSVs, metrics, plots, and report.",
    ),
) -> None:
    comparison_path, metrics_path, plot_path = compare_baselines_pipeline(
        patchcore_predictions=patchcore_predictions,
        dinobank_predictions=dinobank_predictions,
        output_dir=output_dir,
    )
    typer.echo(f"Saved {comparison_path}")
    typer.echo(f"Saved {metrics_path}")
    typer.echo(f"Saved {plot_path}")


@app.command("dinomaly-train")
def dinomaly_train(
    config: Path,
    train_image_dir: Path = typer.Option(
        ...,
        help="Root folder whose child folders are class labels. Normal images train one unified Dinomaly model.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/dinomaly"),
        help="Output directory for Dinomaly artifacts.",
    ),
    validation_image_dir: Path | None = typer.Option(
        None,
        help="Optional validation root with matching class folders.",
    ),
    class_depth: int = typer.Option(
        1,
        min=1,
        help="How many path components under the root form the class label.",
    ),
    class_label: list[str] | None = typer.Option(
        None,
        "--class-label",
        help="Use only this class label. Repeat the option to use multiple classes.",
    ),
) -> None:
    cfg = load_config(config)
    model_path, report_path = train_dinomaly_unified(
        train_image_dir=train_image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        validation_image_dir=validation_image_dir,
        class_labels=class_label,
        config=_dinomaly_config_from_pipeline(cfg),
    )
    typer.echo(f"Saved {model_path}")
    typer.echo(f"Saved {report_path}")


@app.command("dinomaly-validate")
def dinomaly_validate(
    config: Path,
    model: Path,
    validation_image_dir: Path = typer.Option(
        ...,
        help="Validation root with matching class folders.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/dinomaly_validation"),
        help="Output directory for Dinomaly validation artifacts.",
    ),
    class_depth: int | None = typer.Option(
        None,
        min=1,
        help="Override the class_depth saved in the trained model.",
    ),
    class_label: list[str] | None = typer.Option(
        None,
        "--class-label",
        help="Validate only this class label. Repeat the option to validate multiple classes.",
    ),
) -> None:
    load_config(config)
    report_path = validate_dinomaly_unified(
        model_index_path=model,
        validation_image_dir=validation_image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        class_labels=class_label,
    )
    typer.echo(f"Saved {report_path}")


@app.command("dinomaly-predict")
def dinomaly_predict(
    config: Path,
    model: Path,
    image_dir: Path = typer.Option(
        ...,
        help="Blind test image root. Images do not need OK/NG labels.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/dinomaly_blind"),
        help="Output directory for blind prediction CSVs, charts, heatmaps, and OK/NG folders.",
    ),
    class_depth: int = typer.Option(
        1,
        min=1,
        help="How many path components under image_dir form the class label when --class-label is used.",
    ),
    class_label: list[str] | None = typer.Option(
        None,
        "--class-label",
        help="Predict only this class label. Repeat the option to use multiple classes.",
    ),
) -> None:
    load_config(config)
    report_path = predict_dinomaly_blind(
        model_index_path=model,
        image_dir=image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        class_labels=class_label,
    )
    typer.echo(f"Saved {report_path}")


@app.command(hidden=True)
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


@app.command(hidden=True)
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
    class_depth: int | None = typer.Option(
        None,
        min=1,
        help="Override the class_depth saved in the trained dual-branch model.",
    ),
) -> None:
    load_config(config)
    report_path = validate_dual_branch_pipeline(
        model_path=model,
        validation_image_dir=validation_image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
    )
    typer.echo(f"Saved {report_path}")


def _patchcore_config_from_pipeline(
    cfg,
    backbone: str | None = None,
    layers: str | None = None,
    coreset_sampling_ratio: float | None = None,
    num_neighbors: int | None = None,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    image_size: int | None = None,
    center_crop_size: int | None = None,
) -> AnomalibPatchcoreConfig:
    return AnomalibPatchcoreConfig(
        backbone=backbone or cfg.patchcore_backbone,
        layers=tuple((layers.split(",") if layers else cfg.patchcore_layers)),
        coreset_sampling_ratio=(
            cfg.patchcore_coreset_sampling_ratio
            if coreset_sampling_ratio is None
            else coreset_sampling_ratio
        ),
        num_neighbors=cfg.patchcore_num_neighbors if num_neighbors is None else num_neighbors,
        train_batch_size=cfg.patchcore_train_batch_size if train_batch_size is None else train_batch_size,
        eval_batch_size=cfg.patchcore_eval_batch_size if eval_batch_size is None else eval_batch_size,
        num_workers=cfg.patchcore_num_workers if num_workers is None else num_workers,
        image_size=cfg.patchcore_image_size if image_size is None else image_size,
        center_crop_size=cfg.patchcore_center_crop_size if center_crop_size is None else center_crop_size,
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


def _dinobank_config_from_pipeline(cfg) -> DinoBankConfig:
    return DinoBankConfig(
        dinov2_model=cfg.dinov2_model,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        structural_weight=cfg.structural_weight,
        projection_profile_dims=cfg.projection_profile_dims,
        bright_threshold=cfg.bright_threshold,
        edge_threshold=cfg.edge_threshold,
        peak_threshold_std=cfg.peak_threshold_std,
        peak_min_distance=cfg.peak_min_distance,
        structural_image_size=cfg.structural_image_size,
        pca_components=cfg.dinobank_pca_components,
        threshold_quantile=cfg.dinobank_threshold_quantile,
        histogram_bins=cfg.dinobank_histogram_bins,
        montage_samples=cfg.dinobank_montage_samples,
        random_state=cfg.random_state,
    )


def _dinomaly_config_from_pipeline(cfg) -> AnomalibDinomalyConfig:
    return AnomalibDinomalyConfig(
        encoder_name=cfg.dinomaly_encoder_name,
        bottleneck_dropout=cfg.dinomaly_bottleneck_dropout,
        decoder_depth=cfg.dinomaly_decoder_depth,
        target_layers=cfg.dinomaly_target_layers,
        fuse_layer_encoder=cfg.dinomaly_fuse_layer_encoder,
        fuse_layer_decoder=cfg.dinomaly_fuse_layer_decoder,
        remove_class_token=cfg.dinomaly_remove_class_token,
        use_context_recentering=cfg.dinomaly_use_context_recentering,
        train_batch_size=cfg.dinomaly_train_batch_size,
        eval_batch_size=cfg.dinomaly_eval_batch_size,
        num_workers=cfg.dinomaly_num_workers,
        image_size=cfg.dinomaly_image_size,
        crop_size=cfg.dinomaly_crop_size,
        accelerator=cfg.dinomaly_accelerator,
        devices=cfg.dinomaly_devices,
        max_epochs=cfg.dinomaly_max_epochs,
        normal_split_ratio=cfg.dinomaly_normal_split_ratio,
        test_split_ratio=cfg.dinomaly_test_split_ratio,
        val_split_ratio=cfg.dinomaly_val_split_ratio,
        histogram_bins=cfg.dinomaly_histogram_bins,
        montage_samples=cfg.dinomaly_montage_samples,
        heatmap_alpha=cfg.dinomaly_heatmap_alpha,
        seed=cfg.random_state,
    )


def _structural_config_from_pipeline(cfg) -> StructuralFeatureConfig:
    return StructuralFeatureConfig(
        projection_dims=cfg.projection_profile_dims,
        bright_threshold=cfg.bright_threshold,
        edge_threshold=cfg.edge_threshold,
        peak_threshold_std=cfg.peak_threshold_std,
        peak_min_distance=cfg.peak_min_distance,
        image_size=cfg.structural_image_size,
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
