from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile
from typing import Any

import joblib
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

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


@dataclass(frozen=True)
class AnomalibPatchcoreConfig:
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 0
    image_size: int = 256
    center_crop_size: int | None = 224
    accelerator: str = "auto"
    devices: int | str = "auto"
    max_epochs: int = 1
    normal_split_ratio: float = 0.2
    test_split_ratio: float = 0.2
    val_split_ratio: float = 0.5
    montage_samples: int = 30
    histogram_bins: int = 30
    seed: int = 42


def train_patchcore_per_class(
    train_image_dir: Path,
    output_dir: Path,
    class_depth: int = 1,
    validation_image_dir: Path | None = None,
    config: AnomalibPatchcoreConfig | None = None,
) -> tuple[Path, Path]:
    cfg = config or AnomalibPatchcoreConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = list_images(train_image_dir)
    if not train_paths:
        raise ValueError(f"No training images found under {train_image_dir}")

    labels = infer_labels_from_root(train_paths, train_image_dir, class_depth)
    class_labels = sorted(set(labels))
    rows = []
    models = {}
    for label in class_labels:
        class_output_dir = output_dir / "classes" / _safe_filename(label)
        dataset_root = class_output_dir / "dataset"
        _prepare_anomalib_folder_dataset(
            label=label,
            train_image_dir=train_image_dir,
            validation_image_dir=validation_image_dir,
            dataset_root=dataset_root,
            class_depth=class_depth,
        )
        result = _fit_one_anomalib_patchcore(
            label=label,
            dataset_root=dataset_root,
            output_dir=class_output_dir,
            config=cfg,
        )
        rows.append(result)
        models[label] = {
            "label": label,
            "checkpoint_path": result["checkpoint_path"],
            "dataset_root": str(dataset_root),
            "output_dir": str(class_output_dir),
        }

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "patchcore_anomalib_summary.csv"
    summary.to_csv(summary_path, index=False)

    model_index_path = output_dir / "patchcore_models.joblib"
    joblib.dump(
        {
            "backend": "anomalib",
            "config": cfg,
            "class_models": models,
        },
        model_index_path,
    )

    report_path = write_anomalib_patchcore_report(
        summary=summary,
        output_dir=output_dir,
        config=cfg,
        validation_image_dir=validation_image_dir,
    )
    return model_index_path, report_path


def validate_patchcore_per_class(
    model_index_path: Path,
    validation_image_dir: Path,
    output_dir: Path,
    class_depth: int = 1,
    config: AnomalibPatchcoreConfig | None = None,
) -> Path:
    payload = joblib.load(model_index_path)
    cfg = config or payload.get("config") or AnomalibPatchcoreConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, model_info in payload["class_models"].items():
        class_output_dir = output_dir / "classes" / _safe_filename(label)
        dataset_root = class_output_dir / "dataset"
        _prepare_anomalib_folder_dataset(
            label=label,
            train_image_dir=None,
            validation_image_dir=validation_image_dir,
            dataset_root=dataset_root,
            class_depth=class_depth,
        )
        rows.append(
            _test_one_anomalib_patchcore(
                label=label,
                dataset_root=dataset_root,
                checkpoint_path=Path(model_info["checkpoint_path"]),
                output_dir=class_output_dir,
                config=cfg,
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "patchcore_anomalib_validation_summary.csv", index=False)
    report_path = write_anomalib_patchcore_report(
        summary=summary,
        output_dir=output_dir,
        config=cfg,
        validation_image_dir=validation_image_dir,
    )
    return report_path


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


def _prepare_anomalib_folder_dataset(
    label: str,
    train_image_dir: Path | None,
    validation_image_dir: Path | None,
    dataset_root: Path,
    class_depth: int,
) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    train_good_dir = dataset_root / "train" / "good"
    test_good_dir = dataset_root / "test" / "good"
    test_bad_dir = dataset_root / "test" / "abnormal"
    train_good_dir.mkdir(parents=True, exist_ok=True)
    test_good_dir.mkdir(parents=True, exist_ok=True)
    test_bad_dir.mkdir(parents=True, exist_ok=True)

    if train_image_dir is not None:
        train_paths = _paths_for_label(train_image_dir, label, class_depth)
        train_good = [path for path in train_paths if not _has_anomaly_token(path, train_image_dir, class_depth)]
        if not train_good:
            train_good = train_paths
        _link_images(train_good, train_good_dir)

    if validation_image_dir is not None:
        validation_paths = _paths_for_label(validation_image_dir, label, class_depth)
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
        _link_images([path for path in train_good], test_good_dir)

    if train_image_dir is None and not any(test_good_dir.iterdir()) and not any(test_bad_dir.iterdir()):
        raise ValueError(f"No validation images found for label {label}")
    if train_image_dir is not None and not any(train_good_dir.iterdir()):
        raise ValueError(f"No normal training images found for label {label}")


def _paths_for_label(root: Path, label: str, class_depth: int) -> list[Path]:
    paths = []
    for image_path in list_images(root):
        image_label = infer_labels_from_root([image_path], root, class_depth)[0]
        if image_label == label:
            paths.append(image_path)
    return paths


def _has_good_token(path: Path, root: Path, class_depth: int) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    tokens = {part.lower() for part in relative.parts[class_depth:-1]}
    return bool(tokens & GOOD_TOKENS)


def _has_anomaly_token(path: Path, root: Path, class_depth: int) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    tokens = {part.lower() for part in relative.parts[class_depth:-1]}
    return bool(tokens & ANOMALY_TOKENS)


def _link_images(paths: list[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(paths):
        target = target_dir / f"{index:06d}_{source.name}"
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)


def _fit_one_anomalib_patchcore(
    label: str,
    dataset_root: Path,
    output_dir: Path,
    config: AnomalibPatchcoreConfig,
) -> dict[str, Any]:
    Folder, Patchcore, Engine = _load_anomalib_classes()
    datamodule = _make_folder_datamodule(label, dataset_root, config)
    model = _make_patchcore_model(Patchcore, config)
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
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path) if predictions_path else "",
        **_flatten_anomalib_metrics(test_result),
    }


def _test_one_anomalib_patchcore(
    label: str,
    dataset_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    config: AnomalibPatchcoreConfig,
) -> dict[str, Any]:
    Folder, Patchcore, Engine = _load_anomalib_classes()
    datamodule = _make_folder_datamodule(label, dataset_root, config)
    model = _make_patchcore_model(Patchcore, config)
    engine = Engine(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        default_root_dir=output_dir / "anomalib",
    )
    test_result = engine.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    predictions_path = _predict_and_report(engine, model, datamodule, output_dir, config, checkpoint_path)
    return {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path) if predictions_path else "",
        **_flatten_anomalib_metrics(test_result),
    }


def _load_anomalib_classes() -> tuple[Any, Any, Any]:
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as exc:
        raise ImportError(
            "anomalib is required for PatchCore. Install it with `uv sync` after "
            "the pyproject dependency update."
        ) from exc
    return Folder, Patchcore, Engine


def _make_folder_datamodule(label: str, dataset_root: Path, config: AnomalibPatchcoreConfig) -> Any:
    train_good = dataset_root / "train" / "good"
    test_good = dataset_root / "test" / "good"
    test_abnormal = dataset_root / "test" / "abnormal"
    if any(train_good.iterdir()):
        normal_dir = "train/good"
    elif any(test_good.iterdir()):
        normal_dir = "test/good"
    else:
        normal_dir = "test/abnormal"

    kwargs = {
        "name": _safe_filename(label),
        "root": dataset_root,
        "normal_dir": normal_dir,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "num_workers": config.num_workers,
        "normal_split_ratio": config.normal_split_ratio,
        "test_split_mode": "from_dir",
        "test_split_ratio": config.test_split_ratio,
        "val_split_mode": "none",
        "val_split_ratio": config.val_split_ratio,
        "seed": config.seed,
    }
    if any(test_good.iterdir()):
        kwargs["normal_test_dir"] = "test/good"
    if any(test_abnormal.iterdir()):
        kwargs["abnormal_dir"] = "test/abnormal"
    return _instantiate_with_supported_kwargs(_load_anomalib_classes()[0], kwargs)


def _make_patchcore_model(Patchcore: Any, config: AnomalibPatchcoreConfig) -> Any:
    pre_processor = True
    if hasattr(Patchcore, "configure_pre_processor"):
        image_size = (config.image_size, config.image_size)
        center_crop_size = (
            (config.center_crop_size, config.center_crop_size)
            if config.center_crop_size is not None
            else None
        )
        pre_processor = Patchcore.configure_pre_processor(
            image_size=image_size,
            center_crop_size=center_crop_size,
        )
    return Patchcore(
        backbone=config.backbone,
        layers=config.layers,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        num_neighbors=config.num_neighbors,
        pre_processor=pre_processor,
    )


def _instantiate_with_supported_kwargs(cls: Any, kwargs: dict[str, Any]) -> Any:
    import inspect

    signature = inspect.signature(cls)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return cls(**supported)


def _find_checkpoint_path(engine: Any, output_dir: Path) -> Path:
    callback = getattr(getattr(engine, "trainer", None), "checkpoint_callback", None)
    best_model_path = getattr(callback, "best_model_path", "")
    if best_model_path:
        return Path(best_model_path)
    checkpoints = sorted((output_dir / "anomalib").rglob("*.ckpt"))
    if checkpoints:
        return checkpoints[-1]
    return output_dir / "anomalib"


def _predict_and_report(
    engine: Any,
    model: Any,
    datamodule: Any,
    output_dir: Path,
    config: AnomalibPatchcoreConfig,
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
    for batch in [] if predictions is None else predictions:
        rows.extend(_prediction_batch_to_rows(batch))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    path = output_dir / "predictions.csv"
    df.to_csv(path, index=False)
    _plot_prediction_histogram(df, output_dir / "plots", config.histogram_bins)
    _export_prediction_montage(df, output_dir / "montage", config.montage_samples)
    return path


def _prediction_batch_to_rows(batch: Any) -> list[dict[str, Any]]:
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
    rows = []
    for index, image_path in enumerate(paths):
        rows.append(
            {
                "image_path": str(image_path),
                "pred_score": _value_at(scores, index),
                "label": _value_at(labels, index),
                "pred_label": _value_at(pred_labels, index),
            }
        )
    return rows


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, (str, Path)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _value_at(values: list[Any], index: int) -> Any:
    if index >= len(values):
        return None
    value = values[index]
    if hasattr(value, "item"):
        return value.item()
    return value


def _flatten_anomalib_metrics(result: Any) -> dict[str, Any]:
    if isinstance(result, list) and result:
        result = result[0]
    if not isinstance(result, dict):
        return {}
    flattened = {}
    for key, value in result.items():
        if hasattr(value, "item"):
            value = value.item()
        flattened[str(key)] = value
    return flattened


def write_anomalib_patchcore_report(
    summary: pd.DataFrame,
    output_dir: Path,
    config: AnomalibPatchcoreConfig,
    validation_image_dir: Path | None,
) -> Path:
    lines = ["# Anomalib PatchCore Report", ""]
    lines.append("Backend: `anomalib.models.Patchcore`")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    for key, value in config.__dict__.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Classes")
    lines.append("")
    if summary.empty:
        lines.append("No class results were produced.")
    else:
        lines.extend(_dataframe_to_markdown(summary))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `patchcore_models.joblib`: index of per-class anomalib checkpoints",
            "- `patchcore_anomalib_summary.csv`: per-class test metrics and artifact paths",
            "- `classes/*/anomalib`: anomalib trainer output per class",
            "- `classes/*/predictions.csv`: prediction scores when anomalib returns prediction batches",
            "- `classes/*/plots`: score histograms when prediction scores are available",
            "- `classes/*/montage`: highest-score image montage when prediction scores are available",
        ]
    )
    if validation_image_dir is not None:
        lines.append(f"- Validation root: `{validation_image_dir}`")
    report_path = output_dir / "patchcore_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _plot_prediction_histogram(df: pd.DataFrame, output_dir: Path, bins: int) -> None:
    if "pred_score" not in df or df["pred_score"].dropna().empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    pd.to_numeric(df["pred_score"], errors="coerce").dropna().hist(bins=bins)
    plt.title("Anomalib PatchCore prediction scores")
    plt.xlabel("pred_score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_score_histogram.png", dpi=160)
    plt.close()


def _export_prediction_montage(df: pd.DataFrame, output_dir: Path, samples: int) -> None:
    if "pred_score" not in df or "image_path" not in df:
        return
    scored = df.copy()
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
    montage.save(output_dir / "top_prediction_scores.jpg")


def _safe_filename(label: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in label)


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
