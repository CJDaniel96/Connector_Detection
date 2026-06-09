from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "connector_detection_matplotlib"),
)
import matplotlib.pyplot as plt


def compare_baselines(
    patchcore_predictions: Path,
    dinobank_predictions: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    patchcore = _read_predictions(patchcore_predictions, "patchcore")
    dinobank = _read_predictions(dinobank_predictions, "dinobank")
    merged = patchcore.merge(dinobank, on="image_path", how="outer")
    if "ground_truth_patchcore" in merged and "ground_truth_dinobank" in merged:
        merged["ground_truth"] = merged["ground_truth_patchcore"].combine_first(
            merged["ground_truth_dinobank"]
        )
    elif "ground_truth_patchcore" in merged:
        merged["ground_truth"] = merged["ground_truth_patchcore"]
    elif "ground_truth_dinobank" in merged:
        merged["ground_truth"] = merged["ground_truth_dinobank"]

    comparison_path = output_dir / "baseline_comparison.csv"
    merged.to_csv(comparison_path, index=False)

    metrics = _build_metrics(merged)
    metrics_path = output_dir / "baseline_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    plot_path = _plot_scores(merged, output_dir)
    _write_report(output_dir, comparison_path, metrics_path, plot_path, metrics)
    return comparison_path, metrics_path, plot_path


def _read_predictions(path: Path, prefix: str) -> pd.DataFrame:
    csv_paths = _prediction_csv_paths(path)
    if not csv_paths:
        raise ValueError(f"No predictions.csv files found at {path}")
    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if "image_path" not in frame:
            continue
        output = pd.DataFrame({"image_path": frame["image_path"].astype(str)})
        output[f"{prefix}_score"] = pd.to_numeric(
            _first_existing(frame, ["pred_score", "score", "anomaly_score"]),
            errors="coerce",
        )
        if "pred_label" in frame:
            output[f"{prefix}_pred_label"] = frame["pred_label"].map(_normalize_label)
        if "ground_truth" in frame:
            output[f"ground_truth_{prefix}"] = frame["ground_truth"].map(_normalize_label)
        elif "label" in frame:
            parsed = frame["label"].map(_normalize_label)
            if parsed.isin(["good", "ng"]).any():
                output[f"ground_truth_{prefix}"] = parsed
        output[f"{prefix}_source_csv"] = str(csv_path)
        frames.append(output)
    if not frames:
        raise ValueError(f"No usable predictions found at {path}")
    return pd.concat(frames, ignore_index=True)


def _prediction_csv_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("predictions.csv"))


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame:
            return frame[column]
    return pd.Series([np.nan] * len(frame))


def _normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"0", "false", "good", "ok", "normal", "pass"}:
        return "good"
    if text in {"1", "true", "ng", "bad", "defect", "dirty", "foreign", "missing", "shift", "abnormal", "anomaly"}:
        return "ng"
    return text


def _build_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prefix in ("patchcore", "dinobank"):
        score_col = f"{prefix}_score"
        pred_col = f"{prefix}_pred_label"
        if score_col not in merged:
            continue
        row = {
            "baseline": prefix,
            "count": int(merged[score_col].notna().sum()),
            "score_mean": float(pd.to_numeric(merged[score_col], errors="coerce").mean()),
            "score_p95": float(pd.to_numeric(merged[score_col], errors="coerce").quantile(0.95)),
        }
        if "ground_truth" in merged and pred_col in merged:
            scored = merged.dropna(subset=["ground_truth", pred_col])
            scored = scored[scored["ground_truth"].isin(["good", "ng"])]
            if not scored.empty:
                y_true = (scored["ground_truth"] == "ng").astype(int)
                y_pred = (scored[pred_col] == "ng").astype(int)
                row.update(
                    {
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    }
                )
                scores = pd.to_numeric(scored[score_col], errors="coerce")
                if len(set(y_true)) > 1 and scores.notna().all():
                    row["roc_auc"] = float(roc_auc_score(y_true, scores))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_scores(merged: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for prefix in ("patchcore", "dinobank"):
        score_col = f"{prefix}_score"
        if score_col in merged:
            scores = pd.to_numeric(merged[score_col], errors="coerce").dropna()
            if not scores.empty:
                scores.hist(bins=30, alpha=0.5, label=prefix)
    plt.title("Baseline anomaly score distributions")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "baseline_score_histogram.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def _write_report(
    output_dir: Path,
    comparison_path: Path,
    metrics_path: Path,
    plot_path: Path,
    metrics: pd.DataFrame,
) -> Path:
    lines = [
        "# Baseline Comparison Report",
        "",
        f"- Per-image comparison: `{comparison_path}`",
        f"- Metrics: `{metrics_path}`",
        f"- Score histogram: `{plot_path}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(_dataframe_to_markdown(metrics) if not metrics.empty else ["No metrics."])
    report_path = output_dir / "baseline_comparison_report.md"
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
