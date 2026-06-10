from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from connector_detection.images import list_images
from connector_detection.patchcore import (
    ANOMALY_TOKENS,
    GOOD_TOKENS,
    infer_labels_from_root,
    write_anomalib_patchcore_report,
)
from connector_detection.patchcore import (
    _find_checkpoint_path,
    _flatten_anomalib_metrics,
    _hw_tuple,
    _instantiate_with_supported_kwargs,
    _link_images,
    _predict_and_report,
)


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
    _prepare_unified_folder_dataset(
        train_image_dir=None,
        validation_image_dir=validation_image_dir,
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
    summary = pd.DataFrame([result])
    summary.to_csv(output_dir / "dinomaly_anomalib_validation_summary.csv", index=False)
    report_path = write_anomalib_dinomaly_report(
        summary=summary,
        output_dir=output_dir,
        config=cfg,
        validation_image_dir=validation_image_dir,
    )
    return report_path


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
    predictions_path = _predict_and_report(engine, model, datamodule, output_dir, config)
    return {
        "label": "unified",
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path) if predictions_path else "",
        **_flatten_anomalib_metrics(test_result),
    }


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
    predictions_path = _predict_and_report(engine, model, datamodule, output_dir, config, checkpoint_path)
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
    normal_dir = "train/good" if any(train_good.iterdir()) else "test/good"
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
    if any(test_good.iterdir()):
        kwargs["normal_test_dir"] = "test/good"
    if any(test_abnormal.iterdir()):
        kwargs["abnormal_dir"] = "test/abnormal"
    return _instantiate_with_supported_kwargs(_load_anomalib_dinomaly_classes()[0], kwargs)


def _make_dinomaly_model(Dinomaly: Any, config: AnomalibDinomalyConfig) -> Any:
    pre_processor = True
    if hasattr(Dinomaly, "configure_pre_processor"):
        pre_processor = Dinomaly.configure_pre_processor(
            image_size=_hw_tuple(config.image_size) if config.image_size is not None else None,
            crop_size=config.crop_size,
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
    return [
        path
        for path in paths
        if infer_labels_from_root([path], root, class_depth)[0] in wanted
    ]


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
