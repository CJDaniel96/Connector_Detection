"""Validate a trained PatchCore model and generate evaluation charts.

The script:
  1. Runs inference on all images in test_dir (or its good/ / bad/ subdirs).
  2. Applies a threshold (F1-optimal when GT available, or user-specified).
  3. Copies images to output_dir/OK/ and output_dir/NG/.
  4. Saves a CSV with per-image scores.
  5. Generates a 4-panel PNG report: ROC, score distribution, confusion matrix, PR curve.

Supported test_dir layouts
---------------------------
Labelled (full evaluation with AUROC/F1):
    test_dir/
    ├── good/   <- normal images
    └── bad/    <- defective images

Unlabelled (score distribution + threshold sorting only):
    test_dir/
    ├── image1.jpg
    └── ...

Usage (as script):
    uv run scripts/validate_patchcore.py pin_row \\
        --ckpt-path models/patchcore/pin_row/Patchcore/v0/weights/lightning/model.ckpt \\
        --test-dir data/pin_row/test

Usage (as installed command):
    validate-patchcore pin_row --ckpt-path <path>
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

matplotlib.use("Agg")

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

COMPONENTS = ["pin_row", "metal_body"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(test_dir: Path) -> tuple[list[Path], list[int] | None]:
    """Return (paths, gt_labels).  gt_labels is None for flat / unlabelled dirs."""
    good_dir = test_dir / "good"
    bad_dir = test_dir / "bad"

    if good_dir.is_dir() or bad_dir.is_dir():
        paths: list[Path] = []
        labels: list[int] = []
        for p in sorted(good_dir.iterdir()) if good_dir.is_dir() else []:
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
                labels.append(0)
        for p in sorted(bad_dir.iterdir()) if bad_dir.is_dir() else []:
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
                labels.append(1)
        return paths, (labels if labels else None)

    paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return paths, None


def _run_inference(
    model: Patchcore,
    image_dir: Path,
    tmp_dir: Path,
) -> tuple[list[float], list[Path]]:
    """Use Engine.predict for inference. Returns (scores, image_paths)."""
    engine = Engine(default_root_dir=str(tmp_dir))
    dataset = PredictDataset(path=image_dir)

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        return_predictions=True,
    )

    scores: list[float] = []
    paths: list[Path] = []
    if predictions:
        for batch in predictions:
            if batch.pred_score is not None:
                scores.extend(batch.pred_score.cpu().numpy().tolist())
            if batch.image_path is not None:
                paths.extend(Path(p) for p in batch.image_path)

    return scores, paths


def _f1_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    candidates = np.linspace(scores.min(), scores.max(), 300)
    best_t, best_f1 = float(candidates[0]), 0.0
    for t in candidates:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _save_charts(
    scores: np.ndarray,
    gt_labels: np.ndarray | None,
    threshold: float,
    output_dir: Path,
    component: str,
) -> None:
    has_gt = gt_labels is not None and len(np.unique(gt_labels)) == 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PatchCore Validation — {component}", fontsize=14, fontweight="bold")

    # 1. Score Distribution ──────────────────────────────────────────────────
    ax = axes[0, 0]
    if has_gt:
        ok_s = scores[gt_labels == 0]
        ng_s = scores[gt_labels == 1]
        ax.hist(ok_s, bins=40, alpha=0.6, color="steelblue", label=f"OK (n={len(ok_s)})", density=True)
        ax.hist(ng_s, bins=40, alpha=0.6, color="tomato",    label=f"NG (n={len(ng_s)})", density=True)
    else:
        ax.hist(scores, bins=40, color="steelblue", alpha=0.7, label=f"All images (n={len(scores)})", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.4f}")
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend()

    # 2. ROC Curve ────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    if has_gt:
        from sklearn.metrics import roc_auc_score, roc_curve
        fpr, tpr, _ = roc_curve(gt_labels, scores)
        auroc = roc_auc_score(gt_labels, scores)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUROC = {auroc:.4f}")
        ax.plot([0, 1], [0, 1], color="navy", linestyle="--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "Requires labelled test data\n(good/ and bad/ subdirs)",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
        ax.set_title("ROC Curve (N/A — no ground truth)")
        ax.axis("off")

    # 3. Confusion Matrix ─────────────────────────────────────────────────────
    ax = axes[1, 0]
    if has_gt:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score
        preds = (scores >= threshold).astype(int)
        cm = confusion_matrix(gt_labels, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["OK", "NG"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(gt_labels, preds, zero_division=0)
        rec  = recall_score(gt_labels, preds, zero_division=0)
        f1   = f1_score(gt_labels, preds, zero_division=0)
        ax.set_title(f"Confusion Matrix  (threshold = {threshold:.4f})")
        ax.set_xlabel(f"Predicted  |  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    else:
        ax.text(0.5, 0.5, "Requires labelled test data",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
        ax.set_title("Confusion Matrix (N/A)")
        ax.axis("off")

    # 4. Precision-Recall Curve ───────────────────────────────────────────────
    ax = axes[1, 1]
    if has_gt:
        from sklearn.metrics import average_precision_score, precision_recall_curve
        prec_vals, rec_vals, _ = precision_recall_curve(gt_labels, scores)
        ap = average_precision_score(gt_labels, scores)
        ax.plot(rec_vals, prec_vals, color="green", lw=2, label=f"AP = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    else:
        ax.text(0.5, 0.5, "Requires labelled test data",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
        ax.set_title("PR Curve (N/A)")
        ax.axis("off")

    plt.tight_layout()
    out_path = output_dir / f"validation_report_{component}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    typer.echo(f"Charts saved: {out_path}")


def _print_metrics(
    scores: np.ndarray,
    gt_labels: np.ndarray | None,
    threshold: float,
) -> None:
    has_gt = gt_labels is not None and len(np.unique(gt_labels)) == 2
    preds = (scores >= threshold).astype(int)
    typer.echo(f"\n{'─' * 52}")
    typer.echo(f"Threshold      : {threshold:.6f}")
    typer.echo(f"Predicted OK   : {int((preds == 0).sum())}")
    typer.echo(f"Predicted NG   : {int((preds == 1).sum())}")

    if has_gt:
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        typer.echo(f"AUROC          : {roc_auc_score(gt_labels, scores):.4f}")
        typer.echo(f"Avg Precision  : {average_precision_score(gt_labels, scores):.4f}")
        typer.echo(f"Precision      : {precision_score(gt_labels, preds, zero_division=0):.4f}")
        typer.echo(f"Recall         : {recall_score(gt_labels, preds, zero_division=0):.4f}")
        typer.echo(f"F1             : {f1_score(gt_labels, preds, zero_division=0):.4f}")
    typer.echo(f"{'─' * 52}")


# ─────────────────────────────────────────────────────────────────────────────
# Main command
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    component: str = typer.Argument(..., help=f"Component to validate ({' | '.join(COMPONENTS)})"),
    ckpt_path: Path = typer.Option(..., "--ckpt-path", "-k", help="Path to trained .ckpt file"),
    test_dir: Path = typer.Option(None, "--test-dir", "-t", help="Test images directory (default: data/<component>/test)"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Output directory for OK/NG/charts (default: results/<component>)"),
    threshold: float = typer.Option(None, "--threshold", help="Anomaly threshold. If omitted: F1-optimal (needs GT) or 95th-percentile."),
    batch_size: int = typer.Option(32, "--batch-size"),
    num_workers: int = typer.Option(4, "--num-workers"),
) -> None:
    """Validate PatchCore: infer scores, sort images into OK/NG, generate charts."""
    if component not in COMPONENTS:
        typer.echo(f"Error: component must be one of {COMPONENTS}", err=True)
        raise typer.Exit(1)

    test_dir = test_dir or Path(f"data/{component}/test")
    output_dir = output_dir or Path(f"results/{component}")

    if not ckpt_path.exists():
        typer.echo(f"Error: checkpoint not found: {ckpt_path}", err=True)
        raise typer.Exit(1)
    if not test_dir.is_dir():
        typer.echo(f"Error: test directory not found: {test_dir}", err=True)
        raise typer.Exit(1)

    (output_dir / "OK").mkdir(parents=True, exist_ok=True)
    (output_dir / "NG").mkdir(parents=True, exist_ok=True)

    image_paths, gt_labels_list = _collect_images(test_dir)
    if not image_paths:
        typer.echo(f"No images found in {test_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nComponent   : {component}")
    typer.echo(f"Checkpoint  : {ckpt_path}")
    typer.echo(f"Test images : {len(image_paths)}")
    typer.echo(f"Has GT      : {gt_labels_list is not None}")
    typer.echo("\nLoading model and running inference...")

    model = Patchcore.load_from_checkpoint(str(ckpt_path), weights_only=False)
    model.eval()

    tmp_dir = output_dir / "_engine_tmp"
    scores, inferred_paths = _run_inference(model, test_dir, tmp_dir)

    if not inferred_paths:
        inferred_paths = image_paths

    n = min(len(scores), len(inferred_paths))
    scores_arr = np.array(scores[:n])
    paths_final = inferred_paths[:n]
    gt_arr = np.array(gt_labels_list[:n]) if gt_labels_list else None

    # Determine threshold
    if threshold is not None:
        used_threshold = threshold
        typer.echo(f"Using user-specified threshold: {used_threshold:.6f}")
    elif gt_arr is not None and len(np.unique(gt_arr)) == 2:
        used_threshold = _f1_optimal_threshold(scores_arr, gt_arr)
        typer.echo(f"F1-optimal threshold computed: {used_threshold:.6f}")
    else:
        used_threshold = float(np.percentile(scores_arr, 95))
        typer.echo(f"No GT/threshold — using 95th-percentile: {used_threshold:.6f}")

    _print_metrics(scores_arr, gt_arr, used_threshold)

    # Copy images to OK / NG
    typer.echo("\nCopying images to OK/NG directories...")
    ok_count = ng_count = 0
    for img_path, score in zip(paths_final, scores_arr):
        if score >= used_threshold:
            shutil.copy2(img_path, output_dir / "NG" / img_path.name)
            ng_count += 1
        else:
            shutil.copy2(img_path, output_dir / "OK" / img_path.name)
            ok_count += 1

    typer.echo(f"  → OK : {ok_count}  ({output_dir / 'OK'})")
    typer.echo(f"  → NG : {ng_count}  ({output_dir / 'NG'})")

    # Save score CSV
    df = pd.DataFrame({
        "filename": [p.name for p in paths_final],
        "anomaly_score": scores_arr,
        "prediction": ["NG" if s >= used_threshold else "OK" for s in scores_arr],
    })
    if gt_arr is not None:
        df["ground_truth"] = ["NG" if g == 1 else "OK" for g in gt_arr]
    csv_path = output_dir / f"scores_{component}.csv"
    df.to_csv(csv_path, index=False)
    typer.echo(f"\nScore CSV   : {csv_path}")

    _save_charts(scores_arr, gt_arr, used_threshold, output_dir, component)

    # Cleanup
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
