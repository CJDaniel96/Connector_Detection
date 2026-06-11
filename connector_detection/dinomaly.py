from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from connector_detection.images import list_images
from connector_detection.patchcore import (
    ANOMALY_TOKENS,
    GOOD_TOKENS,
    infer_labels_from_root,
    write_anomalib_patchcore_report,
)
from connector_detection.patchcore import (
    _first_present,
    _find_checkpoint_path,
    _flatten_anomalib_metrics,
    _hw_tuple,
    _instantiate_with_supported_kwargs,
    _link_images,
    _to_list,
    _value_at,
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "connector_detection_matplotlib"),
)
import matplotlib.pyplot as plt

from connector_detection.evaluation import finalize_evaluation_outputs, has_label_tokens


@dataclass(frozen=True)
class AnomalibDinomalyConfig:
    encoder_name: str = "dinov2reg_vit_base_14"
    bottleneck_dropout: float = 0.2
    decoder_depth: int = 8
    target_layers: tuple[int, ...] | None = None
    fuse_layer_encoder: tuple[tuple[int, ...], ...] | None = None
    fuse_layer_decoder: tuple[tuple[int, ...], ...] | None = None
    remove_class_token: bool = False
    use_context_recentering: bool = False
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 0
    image_size: int | tuple[int, int] | None = (448, 448)
    crop_size: int | None = 392
    accelerator: str = "auto"
    devices: int | str = "auto"
    max_epochs: int = 10
    normal_split_ratio: float = 0.2
    test_split_ratio: float = 0.2
    val_split_ratio: float = 0.5
    histogram_bins: int = 30
    montage_samples: int = 30
    heatmap_alpha: float = 0.45
    seed: int = 42


def train_dinomaly_unified(
    train_image_dir: Path,
    output_dir: Path,
    class_depth: int = 1,
    validation_image_dir: Path | None = None,
    class_labels: list[str] | None = None,
    config: AnomalibDinomalyConfig | None = None,
) -> tuple[Path, Path]:
    cfg = config or AnomalibDinomalyConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = output_dir / "dataset"
    train_paths = _normal_training_paths(train_image_dir, class_depth, class_labels)
    if not train_paths:
        raise ValueError(f"No normal training images found under {train_image_dir}")

    _prepare_unified_folder_dataset(
        train_image_dir=train_image_dir,
        validation_image_dir=validation_image_dir,
        dataset_root=dataset_root,
        class_depth=class_depth,
        class_labels=class_labels,
    )
    result = _fit_one_dinomaly(dataset_root=dataset_root, output_dir=output_dir, config=cfg)
    summary = pd.DataFrame([result])
    summary_path = output_dir / "dinomaly_anomalib_summary.csv"
    summary.to_csv(summary_path, index=False)

    model_index_path = output_dir / "dinomaly_model.joblib"
    labels = sorted(set(infer_labels_from_root(train_paths, train_image_dir, class_depth)))
    joblib.dump(
        {
            "backend": "anomalib",
            "model_name": "Dinomaly",
            "config": cfg,
            "class_depth": class_depth,
            "class_labels": labels,
            "checkpoint_path": result["checkpoint_path"],
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
        },
        model_index_path,
    )
    report_path = write_anomalib_dinomaly_report(
        summary=summary,
        output_dir=output_dir,
        config=cfg,
        validation_image_dir=validation_image_dir,
    )
    return model_index_path, report_path


def validate_dinomaly_unified(
    model_index_path: Path,
    validation_image_dir: Path,
    output_dir: Path,
    class_depth: int | None = None,
    class_labels: list[str] | None = None,
    config: AnomalibDinomalyConfig | None = None,
) -> Path:
    return evaluate_dinomaly_unified(
        model_index_path=model_index_path,
        image_dir=validation_image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        class_labels=class_labels,
        config=config,
    )


def evaluate_dinomaly_unified(
    model_index_path: Path,
    image_dir: Path,
    output_dir: Path,
    class_depth: int | None = None,
    class_labels: list[str] | None = None,
    config: AnomalibDinomalyConfig | None = None,
) -> Path:
    payload = joblib.load(model_index_path)
    cfg = config or payload.get("config") or AnomalibDinomalyConfig()
    resolved_class_depth = class_depth or int(payload.get("class_depth", 1))
    known_labels = set(payload.get("class_labels", []))
    requested = set(class_labels or [])
    missing = sorted(requested - known_labels)
    if missing:
        raise ValueError(
            f"Unknown Dinomaly class label(s): {', '.join(missing)}. "
            f"Available labels: {', '.join(sorted(known_labels))}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = output_dir / "dataset"
    image_paths = _filter_class_labels(
        list_images(image_dir),
        image_dir,
        resolved_class_depth,
        class_labels,
    )
    if not image_paths:
        raise ValueError(f"No evaluation images found under {image_dir}")
    is_labeled = has_label_tokens(image_paths, image_dir, resolved_class_depth)
    if is_labeled:
        _prepare_unified_folder_dataset(
            train_image_dir=None,
            validation_image_dir=image_dir,
            dataset_root=dataset_root,
            class_depth=resolved_class_depth,
            class_labels=class_labels,
        )
        result = _test_one_dinomaly(
            dataset_root=dataset_root,
            checkpoint_path=Path(payload["checkpoint_path"]),
            output_dir=output_dir,
            config=cfg,
        )
    else:
        _prepare_blind_folder_dataset(
            image_dir=image_dir,
            dataset_root=dataset_root,
            class_depth=resolved_class_depth,
            class_labels=class_labels,
        )
        predictions_path = _predict_one_dinomaly(
            dataset_root=dataset_root,
            checkpoint_path=Path(payload["checkpoint_path"]),
            output_dir=output_dir,
            config=cfg,
        )
        result = {
            "label": "unified",
            "checkpoint_path": str(payload["checkpoint_path"]),
            "predictions_path": str(predictions_path) if predictions_path else "",
        }

    summary = pd.DataFrame([result])
    summary.to_csv(output_dir / "dinomaly_anomalib_evaluation_summary.csv", index=False)
    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.exists():
        raise RuntimeError("Dinomaly did not return prediction rows for evaluation data.")
    predictions = pd.read_csv(predictions_path)
    _, evaluation_report = finalize_evaluation_outputs(
        predictions=predictions,
        output_dir=output_dir,
        method_name="Dinomaly",
        image_root=image_dir,
        class_depth=resolved_class_depth,
        histogram_bins=cfg.histogram_bins,
        montage_samples=cfg.montage_samples,
    )
    write_anomalib_dinomaly_report(
        summary=summary,
        output_dir=output_dir,
        config=cfg,
        validation_image_dir=image_dir if is_labeled else None,
    )
    return evaluation_report


def predict_dinomaly_blind(
    model_index_path: Path,
    image_dir: Path,
    output_dir: Path,
    class_depth: int = 1,
    class_labels: list[str] | None = None,
    config: AnomalibDinomalyConfig | None = None,
) -> Path:
    return evaluate_dinomaly_unified(
        model_index_path=model_index_path,
        image_dir=image_dir,
        output_dir=output_dir,
        class_depth=class_depth,
        class_labels=class_labels,
        config=config,
    )


def _prepare_unified_folder_dataset(
    train_image_dir: Path | None,
    validation_image_dir: Path | None,
    dataset_root: Path,
    class_depth: int,
    class_labels: list[str] | None,
) -> None:
    if dataset_root.exists():
        import shutil

        shutil.rmtree(dataset_root)
    train_good_dir = dataset_root / "train" / "good"
    test_good_dir = dataset_root / "test" / "good"
    test_bad_dir = dataset_root / "test" / "abnormal"
    train_good_dir.mkdir(parents=True, exist_ok=True)
    test_good_dir.mkdir(parents=True, exist_ok=True)
    test_bad_dir.mkdir(parents=True, exist_ok=True)

    train_good: list[Path] = []
    if train_image_dir is not None:
        train_good = _normal_training_paths(train_image_dir, class_depth, class_labels)
        _link_images(train_good, train_good_dir)

    if validation_image_dir is not None:
        validation_paths = _filter_class_labels(
            list_images(validation_image_dir),
            validation_image_dir,
            class_depth,
            class_labels,
        )
        normal_paths = [
            path
            for path in validation_paths
            if _has_good_token(path, validation_image_dir, class_depth)
            or not _has_anomaly_token(path, validation_image_dir, class_depth)
        ]
        abnormal_paths = [
            path
            for path in validation_paths
            if _has_anomaly_token(path, validation_image_dir, class_depth)
        ]
        _link_images(normal_paths, test_good_dir)
        _link_images(abnormal_paths, test_bad_dir)

    if train_image_dir is not None and not any(test_good_dir.iterdir()) and not any(test_bad_dir.iterdir()):
        _link_images(train_good, test_good_dir)

    if train_image_dir is None and not any(test_good_dir.iterdir()) and not any(test_bad_dir.iterdir()):
        raise ValueError("No validation images found for Dinomaly validation")
    if train_image_dir is not None and not any(train_good_dir.iterdir()):
        raise ValueError("No normal training images found for Dinomaly training")


def _prepare_blind_folder_dataset(
    image_dir: Path,
    dataset_root: Path,
    class_depth: int,
    class_labels: list[str] | None,
) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    train_good_dir = dataset_root / "train" / "good"
    test_good_dir = dataset_root / "test" / "good"
    train_good_dir.mkdir(parents=True, exist_ok=True)
    test_good_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)
    if class_labels:
        image_paths = _filter_class_labels(image_paths, image_dir, class_depth, class_labels)
    if not image_paths:
        raise ValueError(f"No blind test images found under {image_dir}")
    _link_images(image_paths, test_good_dir)


def _fit_one_dinomaly(
    dataset_root: Path,
    output_dir: Path,
    config: AnomalibDinomalyConfig,
) -> dict[str, Any]:
    Folder, Dinomaly, Engine = _load_anomalib_dinomaly_classes()
    datamodule = _make_folder_datamodule(dataset_root, config)
    model = _make_dinomaly_model(Dinomaly, config)
    engine = Engine(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        default_root_dir=output_dir / "anomalib",
    )
    engine.fit(model=model, datamodule=datamodule)
    test_result = engine.test(model=model, datamodule=datamodule)
    checkpoint_path = _find_checkpoint_path(engine, output_dir)
    predictions_path = _predict_and_visualize(engine, model, datamodule, output_dir, config)
    return {
        "label": "unified",
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path) if predictions_path else "",
        **_flatten_anomalib_metrics(test_result),
    }


def _predict_one_dinomaly(
    dataset_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    config: AnomalibDinomalyConfig,
) -> Path | None:
    Folder, Dinomaly, Engine = _load_anomalib_dinomaly_classes()
    datamodule = _make_folder_datamodule(dataset_root, config)
    model = _make_dinomaly_model(Dinomaly, config)
    engine = Engine(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        default_root_dir=output_dir / "anomalib",
    )
    return _predict_and_visualize(
        engine,
        model,
        datamodule,
        output_dir,
        config,
        checkpoint_path,
    )


def _test_one_dinomaly(
    dataset_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    config: AnomalibDinomalyConfig,
) -> dict[str, Any]:
    Folder, Dinomaly, Engine = _load_anomalib_dinomaly_classes()
    datamodule = _make_folder_datamodule(dataset_root, config)
    model = _make_dinomaly_model(Dinomaly, config)
    engine = Engine(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        default_root_dir=output_dir / "anomalib",
    )
    test_result = engine.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    predictions_path = _predict_and_visualize(
        engine,
        model,
        datamodule,
        output_dir,
        config,
        checkpoint_path,
    )
    return {
        "label": "unified",
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path) if predictions_path else "",
        **_flatten_anomalib_metrics(test_result),
    }


def _load_anomalib_dinomaly_classes() -> tuple[Any, Any, Any]:
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Dinomaly
    except ImportError as exc:
        raise ImportError(
            "anomalib>=2.1 is required for Dinomaly. Install dependencies with `uv sync`."
        ) from exc
    return Folder, Dinomaly, Engine


def _make_folder_datamodule(dataset_root: Path, config: AnomalibDinomalyConfig) -> Any:
    train_good = dataset_root / "train" / "good"
    test_good = dataset_root / "test" / "good"
    test_abnormal = dataset_root / "test" / "abnormal"
    has_train_good = train_good.exists() and any(train_good.iterdir())
    has_test_good = test_good.exists() and any(test_good.iterdir())
    has_test_abnormal = test_abnormal.exists() and any(test_abnormal.iterdir())
    normal_dir = "train/good" if has_train_good else "test/good"
    kwargs = {
        "name": "dinomaly_unified",
        "root": dataset_root,
        "normal_dir": normal_dir,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "num_workers": config.num_workers,
        "normal_split_ratio": config.normal_split_ratio,
        "test_split_mode": "from_dir",
        "test_split_ratio": config.test_split_ratio,
        "val_split_mode": "same_as_test",
        "val_split_ratio": config.val_split_ratio,
        "seed": config.seed,
    }
    if has_test_good:
        kwargs["normal_test_dir"] = "test/good"
    if has_test_abnormal:
        kwargs["abnormal_dir"] = "test/abnormal"
    return _instantiate_with_supported_kwargs(_load_anomalib_dinomaly_classes()[0], kwargs)


def _make_dinomaly_model(Dinomaly: Any, config: AnomalibDinomalyConfig) -> Any:
    pre_processor = True
    if hasattr(Dinomaly, "configure_pre_processor"):
        image_size = _hw_tuple(config.image_size) if config.image_size is not None else None
        pre_processor = (
            _make_no_crop_pre_processor(image_size)
            if config.crop_size is None
            else Dinomaly.configure_pre_processor(
                image_size=image_size,
                crop_size=config.crop_size,
            )
        )
    return Dinomaly(
        encoder_name=config.encoder_name,
        bottleneck_dropout=config.bottleneck_dropout,
        decoder_depth=config.decoder_depth,
        target_layers=list(config.target_layers) if config.target_layers is not None else None,
        fuse_layer_encoder=(
            [list(group) for group in config.fuse_layer_encoder]
            if config.fuse_layer_encoder is not None
            else None
        ),
        fuse_layer_decoder=(
            [list(group) for group in config.fuse_layer_decoder]
            if config.fuse_layer_decoder is not None
            else None
        ),
        remove_class_token=config.remove_class_token,
        use_context_recentering=config.use_context_recentering,
        pre_processor=pre_processor,
    )


def _make_no_crop_pre_processor(image_size: tuple[int, int] | None) -> Any:
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Compose, Normalize, Resize

    transforms = []
    if image_size is not None:
        transforms.append(Resize(image_size))
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return PreProcessor(transform=Compose(transforms))


def _predict_and_visualize(
    engine: Any,
    model: Any,
    datamodule: Any,
    output_dir: Path,
    config: AnomalibDinomalyConfig,
    checkpoint_path: Path | None = None,
) -> Path | None:
    try:
        predictions = engine.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
        )
    except Exception:
        return None

    rows = []
    heatmap_dir = output_dir / "heatmaps"
    heatmap_alpha = getattr(config, "heatmap_alpha", 0.45)
    for batch in [] if predictions is None else predictions:
        rows.extend(_prediction_batch_to_rows(batch, heatmap_dir, heatmap_alpha))
    if not rows:
        return None

    df = pd.DataFrame(rows)
    path = output_dir / "predictions.csv"
    df.to_csv(path, index=False)
    _plot_score_histogram(df, output_dir / "plots", getattr(config, "histogram_bins", 30))
    _export_top_score_montages(df, output_dir / "montage", getattr(config, "montage_samples", 30))
    return path


def _prediction_batch_to_rows(
    batch: Any,
    heatmap_dir: Path,
    alpha: float,
) -> list[dict[str, Any]]:
    if isinstance(batch, dict):
        data = batch
    elif hasattr(batch, "__dict__"):
        data = batch.__dict__
    else:
        return []

    paths = _to_list(_first_present(data, ("image_path", "path", "image_paths")))
    scores = _to_list(_first_present(data, ("pred_score", "anomaly_score", "score")))
    labels = _to_list(_first_present(data, ("label", "gt_label")))
    pred_labels = _to_list(data.get("pred_label"))
    anomaly_maps = _to_list(_first_present(data, ("anomaly_map", "pred_mask", "heatmap")))

    rows = []
    for index, image_path in enumerate(paths):
        raw_heatmap_path = None
        heatmap_path = None
        overlay_path = None
        anomaly_map = _value_at(anomaly_maps, index)
        if anomaly_map is not None:
            raw_heatmap_path, heatmap_path, overlay_path = _save_heatmap_outputs(
                image_path=Path(image_path),
                anomaly_map=anomaly_map,
                output_dir=heatmap_dir,
                index=index,
                alpha=alpha,
            )
        rows.append(
            {
                "image_path": str(image_path),
                "pred_score": _value_at(scores, index),
                "label": _value_at(labels, index),
                "pred_label": _value_at(pred_labels, index),
                "raw_heatmap_path": str(raw_heatmap_path) if raw_heatmap_path else "",
                "heatmap_path": str(heatmap_path) if heatmap_path else "",
                "overlay_path": str(overlay_path) if overlay_path else "",
            }
        )
    return rows


def _save_heatmap_outputs(
    image_path: Path,
    anomaly_map: Any,
    output_dir: Path,
    index: int,
    alpha: float,
) -> tuple[Path, Path, Path] | tuple[None, None, None]:
    heatmap = _anomaly_map_to_array(anomaly_map)
    if heatmap is None:
        return None, None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}_{index:04d}"
    raw_heatmap_path = output_dir / f"{stem}_raw_heatmap.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    colored = _visualize_heatmap(heatmap)
    colored.save(raw_heatmap_path)

    if image_path.exists():
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            overlay = _overlay_heatmap(image, colored, alpha)
            overlay.save(heatmap_path)
            overlay.save(overlay_path)
    else:
        colored.save(heatmap_path)
        colored.save(overlay_path)
    return raw_heatmap_path, heatmap_path, overlay_path


def _anomaly_map_to_array(anomaly_map: Any) -> np.ndarray | None:
    if hasattr(anomaly_map, "detach"):
        anomaly_map = anomaly_map.detach().cpu().numpy()
    array = np.asarray(anomaly_map)
    if array.size == 0:
        return None
    array = np.squeeze(array)
    if array.ndim == 3:
        array = array[0] if array.shape[0] in (1, 3) else array[:, :, 0]
    if array.ndim != 2:
        return None
    array = array.astype(np.float32)
    array = array - float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if max_value > 0:
        array = array / max_value
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)


def _visualize_heatmap(heatmap: np.ndarray) -> Image.Image:
    try:
        from anomalib.visualization.image.functional import visualize_anomaly_map

        return visualize_anomaly_map(heatmap, normalize=True, colormap=True).convert("RGB")
    except Exception:
        cmap = plt.get_cmap("jet")
        rgba = cmap(np.clip(heatmap, 0.0, 1.0))
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")


def _overlay_heatmap(image: Image.Image, heatmap: Image.Image, alpha: float) -> Image.Image:
    heatmap = heatmap.resize(image.size, Image.Resampling.BILINEAR)
    try:
        from anomalib.visualization.image.functional import overlay_images

        return overlay_images(image, heatmap, alpha=alpha).convert("RGB")
    except Exception:
        return Image.blend(image, heatmap, alpha=alpha)


def _plot_score_histogram(df: pd.DataFrame, output_dir: Path, bins: int) -> None:
    if "pred_score" not in df:
        return
    scores = pd.to_numeric(df["pred_score"], errors="coerce")
    if scores.dropna().empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if "label" in df and df["label"].notna().any():
        for label, group in df.groupby("label"):
            pd.to_numeric(group["pred_score"], errors="coerce").dropna().hist(
                bins=bins,
                alpha=0.55,
                label=str(label),
            )
        plt.legend()
    else:
        scores.dropna().hist(bins=bins)
    plt.title("Dinomaly anomaly scores")
    plt.xlabel("pred_score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "score_histogram.png", dpi=160)
    plt.close()


def _export_top_score_montages(df: pd.DataFrame, output_dir: Path, samples: int) -> None:
    if "pred_score" not in df or "image_path" not in df:
        return
    scored = df.copy()
    scored["pred_score"] = pd.to_numeric(scored["pred_score"], errors="coerce")
    scored = scored.dropna(subset=["pred_score"]).sort_values("pred_score", ascending=False)
    if scored.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = scored.head(samples)
    _save_montage(scored, "image_path", output_dir / "top_anomaly_scores.jpg")
    if "overlay_path" in scored and scored["overlay_path"].astype(str).str.len().gt(0).any():
        overlay_scored = scored[scored["overlay_path"].astype(str).str.len() > 0]
        _save_montage(overlay_scored, "overlay_path", output_dir / "top_anomaly_overlays.jpg")


def _save_montage(df: pd.DataFrame, path_column: str, output_path: Path) -> None:
    tile_size = (180, 120)
    columns = 5
    rows = max(1, (len(df) + columns - 1) // columns)
    montage = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(montage)
    for index, row in enumerate(df.itertuples(index=False)):
        image_path = Path(getattr(row, path_column))
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
    montage.save(output_path)


def _add_blind_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    output["source_image_path"] = output["image_path"].map(lambda value: str(Path(value).resolve()))
    output["prediction"] = [
        _prediction_bucket(label)
        for label in output.get("pred_label", pd.Series([None] * len(output)))
    ]
    return output


def _prediction_bucket(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().lower()
    if text in {"0", "false", "good", "normal", "ok", "pass"}:
        return "OK"
    if text in {"1", "true", "ng", "bad", "abnormal", "anomaly", "anomalous", "defect"}:
        return "NG"
    return "NG" if text not in {"none", ""} else "UNKNOWN"


def _copy_blind_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    for bucket in ("OK", "NG", "UNKNOWN"):
        (output_dir / bucket).mkdir(parents=True, exist_ok=True)
        (output_dir / f"{bucket}_overlays").mkdir(parents=True, exist_ok=True)

    for row in predictions.itertuples(index=False):
        bucket = str(row.prediction)
        source = Path(row.source_image_path)
        if source.exists():
            _copy_image(source, output_dir / bucket)
        overlay_value = str(getattr(row, "overlay_path", "")).strip()
        overlay_path = Path(overlay_value) if overlay_value else None
        if overlay_path is not None and overlay_path.is_file():
            _copy_image(overlay_path, output_dir / f"{bucket}_overlays")


def _copy_image(source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name
    if target.exists():
        target = output_dir / f"{source.stem}_{abs(hash(source))}{source.suffix}"
    shutil.copy2(source, target)


def _plot_blind_summary(predictions: pd.DataFrame, output_dir: Path, bins: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = predictions["prediction"].value_counts().reindex(["OK", "NG", "UNKNOWN"], fill_value=0)

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#3b7d4f", "#b73737", "#777777"])
    plt.title("Dinomaly blind prediction counts")
    plt.xlabel("prediction")
    plt.ylabel("count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_counts.png", dpi=160)
    plt.close()

    if "pred_score" in predictions:
        scores = pd.to_numeric(predictions["pred_score"], errors="coerce")
        if not scores.dropna().empty:
            plt.figure(figsize=(8, 5))
            for bucket, group in predictions.groupby("prediction"):
                pd.to_numeric(group["pred_score"], errors="coerce").dropna().hist(
                    bins=bins,
                    alpha=0.55,
                    label=str(bucket),
                )
            plt.title("Dinomaly blind anomaly scores")
            plt.xlabel("pred_score")
            plt.ylabel("count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "score_histogram_by_prediction.png", dpi=160)
            plt.close()

            sorted_scores = predictions.copy()
            sorted_scores["pred_score"] = scores
            sorted_scores = sorted_scores.dropna(subset=["pred_score"]).sort_values(
                "pred_score",
                ascending=False,
            )
            top = sorted_scores.head(min(50, len(sorted_scores)))
            if not top.empty:
                plt.figure(figsize=(10, max(4, min(12, len(top) * 0.24))))
                plt.barh(
                    [Path(path).name for path in top["source_image_path"]][::-1],
                    top["pred_score"].to_numpy()[::-1],
                    color="#b73737",
                )
                plt.title("Top Dinomaly anomaly scores")
                plt.xlabel("pred_score")
                plt.tight_layout()
                plt.savefig(output_dir / "top_anomaly_scores.png", dpi=160)
                plt.close()


def _write_blind_report(
    output_dir: Path,
    predictions_path: Path,
    predictions: pd.DataFrame,
) -> Path:
    counts = predictions["prediction"].value_counts().reindex(["OK", "NG", "UNKNOWN"], fill_value=0)
    lines = [
        "# Dinomaly Blind Prediction Report",
        "",
        f"- Predictions: `{predictions_path}`",
        f"- Total images: {len(predictions)}",
        f"- OK: {int(counts['OK'])}",
        f"- NG: {int(counts['NG'])}",
        f"- UNKNOWN: {int(counts['UNKNOWN'])}",
        "",
        "## Artifacts",
        "",
        "- `classified/OK`: original images predicted OK",
        "- `classified/NG`: original images predicted NG",
        "- `classified/*_overlays`: copied heatmap overlays by prediction bucket",
        "- `heatmaps`: raw and overlay anomaly maps",
        "- `analysis/prediction_counts.png`: OK/NG count chart",
        "- `analysis/score_histogram_by_prediction.png`: score distribution by prediction",
        "- `analysis/top_anomaly_scores.png`: top anomaly score bar chart",
        "- `montage/top_anomaly_scores.jpg`: highest-score input montage",
        "- `montage/top_anomaly_overlays.jpg`: highest-score overlay montage",
    ]
    report_path = output_dir / "blind_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _normal_training_paths(
    root: Path,
    class_depth: int,
    class_labels: list[str] | None,
) -> list[Path]:
    paths = [
        path
        for path in list_images(root)
        if not _has_anomaly_token(path, root, class_depth)
    ]
    return _filter_class_labels(paths, root, class_depth, class_labels)


def _filter_class_labels(
    paths: list[Path],
    root: Path,
    class_depth: int,
    class_labels: list[str] | None,
) -> list[Path]:
    if not class_labels:
        return paths
    wanted = set(class_labels)
    selected = []
    for path in paths:
        try:
            label = infer_labels_from_root([path], root, class_depth)[0]
        except ValueError:
            if len(wanted) == 1:
                selected.append(path)
            continue
        if label in wanted:
            selected.append(path)
    return selected


def _has_good_token(path: Path, root: Path, class_depth: int) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    tokens = {part.lower() for part in relative.parts[class_depth:-1]}
    return bool(tokens & GOOD_TOKENS)


def _has_anomaly_token(path: Path, root: Path, class_depth: int) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    tokens = {part.lower() for part in relative.parts[class_depth:-1]}
    return bool(tokens & ANOMALY_TOKENS)


def write_anomalib_dinomaly_report(
    summary: pd.DataFrame,
    output_dir: Path,
    config: AnomalibDinomalyConfig,
    validation_image_dir: Path | None,
) -> Path:
    lines = ["# Anomalib Dinomaly Report", ""]
    lines.append("Backend: `anomalib.models.Dinomaly`")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    for key, value in config.__dict__.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    if summary.empty:
        lines.append("No results were produced.")
    else:
        lines.extend(_dataframe_to_markdown(summary))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `dinomaly_model.joblib`: checkpoint index and training config",
            "- `dinomaly_anomalib_summary.csv`: anomalib test metrics and artifact paths",
            "- `anomalib`: anomalib trainer logs, checkpoints, and visual artifacts",
            "- `predictions.csv`: prediction scores when anomalib returns prediction batches",
            "- `heatmaps/*_raw_heatmap.png`: raw anomaly heatmaps",
            "- `heatmaps/*_overlay.png`: heatmaps overlaid on input images",
            "- `plots/score_histogram.png`: anomaly score histogram",
            "- `montage/top_anomaly_scores.jpg`: highest-score input images",
            "- `montage/top_anomaly_overlays.jpg`: highest-score heatmap overlays",
        ]
    )
    if validation_image_dir is not None:
        lines.append(f"- Validation root: `{validation_image_dir}`")
    report_path = output_dir / "dinomaly_report.md"
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
