"""Train a PatchCore anomaly detection model for connector parts.

PatchCore is a memory-bank method: only normal (good) images are needed for
training. Feature extraction runs in a single epoch. The resulting coreset is
stored as the memory bank and used at inference time.

Usage (as script):
    uv run scripts/train_patchcore.py pin_row
    uv run scripts/train_patchcore.py metal_body --config configs/patchcore_body.yaml

Usage (as installed command):
    train-patchcore pin_row
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from anomalib.data import Folder
from anomalib.data.utils.split import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

COMPONENTS = ["pin_row", "metal_body"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _has_images(path: Path) -> bool:
    return path.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in path.iterdir())


@app.command()
def main(
    component: str = typer.Argument(..., help=f"Component to train ({' | '.join(COMPONENTS)})"),
    data_dir: Path = typer.Option(None, "--data-dir", "-d", help="Component data root (default: data/<component>)"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Checkpoint output directory (default: models/patchcore/<component>)"),
    config: Path = typer.Option(None, "--config", "-c", help="YAML config (default: configs/patchcore_<component>.yaml)"),
) -> None:
    """Train PatchCore on normal images and optionally evaluate on test data."""
    if component not in COMPONENTS:
        typer.echo(f"Error: component must be one of {COMPONENTS}", err=True)
        raise typer.Exit(1)

    data_dir = data_dir or Path(f"data/{component}")
    output_dir = output_dir or Path(f"models/patchcore/{component}")
    config_path = config or Path(f"configs/patchcore_{component}.yaml")

    train_good = data_dir / "train" / "good"
    if not _has_images(train_good):
        typer.echo(
            f"Error: no training images in {train_good}\n"
            "Run normalize-orientation first to prepare training data.",
            err=True,
        )
        raise typer.Exit(1)

    cfg: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        typer.echo(f"Config loaded: {config_path}")
    else:
        typer.echo(f"Config not found at {config_path} — using defaults.")

    input_size: tuple[int, int] = tuple(cfg.get("input_size", [256, 256]))

    test_bad = data_dir / "test" / "bad"
    test_good = data_dir / "test" / "good"
    has_abnormal = _has_images(test_bad)
    has_normal_test = _has_images(test_good)

    if has_abnormal and has_normal_test:
        test_split_mode = TestSplitMode.FROM_DIR
        val_split_mode = ValSplitMode.FROM_TEST
        abnormal_dir = "test/bad"
        normal_test_dir = "test/good"
        typer.echo("Test split: using test/good + test/bad.")
    elif has_normal_test:
        test_split_mode = TestSplitMode.FROM_DIR
        val_split_mode = ValSplitMode.FROM_TEST
        abnormal_dir = None
        normal_test_dir = "test/good"
        typer.echo("Test split: using test/good (no abnormal images yet).")
    else:
        # No test data yet — split 20% from train/good for validation
        test_split_mode = TestSplitMode.FROM_DIR
        val_split_mode = ValSplitMode.FROM_TEST
        abnormal_dir = None
        normal_test_dir = None
        typer.echo("No test data found — will split 20% of train/good for validation.")

    datamodule = Folder(
        name=component,
        root=data_dir,
        normal_dir="train/good",
        abnormal_dir=abnormal_dir,
        normal_test_dir=normal_test_dir,
        train_batch_size=cfg.get("train_batch_size", 32),
        eval_batch_size=cfg.get("eval_batch_size", 32),
        num_workers=cfg.get("num_workers", 4),
        test_split_mode=test_split_mode,
        val_split_mode=val_split_mode,
    )

    pre_processor = Patchcore.configure_pre_processor(image_size=input_size)
    model = Patchcore(
        backbone=cfg.get("backbone", "wide_resnet50_2"),
        layers=cfg.get("layers", ["layer2", "layer3"]),
        pre_trained=cfg.get("pre_trained", True),
        coreset_sampling_ratio=cfg.get("coreset_sampling_ratio", 0.1),
        num_neighbors=cfg.get("num_neighbors", 9),
        pre_processor=pre_processor,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(default_root_dir=str(output_dir))

    typer.echo(f"\n{'─' * 52}")
    typer.echo(f"Component  : {component}")
    typer.echo(f"Data dir   : {data_dir}")
    typer.echo(f"Output dir : {output_dir}")
    typer.echo(f"Backbone   : {cfg.get('backbone', 'wide_resnet50_2')}")
    typer.echo(f"Layers     : {cfg.get('layers', ['layer2', 'layer3'])}")
    typer.echo(f"Input size : {input_size}")
    typer.echo(f"Coreset    : {cfg.get('coreset_sampling_ratio', 0.1)}")
    typer.echo(f"{'─' * 52}\n")

    engine.train(model=model, datamodule=datamodule)

    if has_abnormal or has_normal_test:
        typer.echo("\nRunning test evaluation to calibrate anomaly threshold...")
        engine.test(model=model, datamodule=datamodule)

    ckpt_candidates = sorted(output_dir.rglob("*.ckpt"))
    if ckpt_candidates:
        best = ckpt_candidates[-1]
        typer.echo(f"\nCheckpoint: {best}")
    else:
        typer.echo(f"\nTraining complete. Checkpoints are in: {output_dir}")
