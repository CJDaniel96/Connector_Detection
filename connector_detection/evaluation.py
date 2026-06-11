from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

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

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "connector_detection_matplotlib"),
)
import matplotlib.pyplot as plt


def has_label_tokens(image_paths: list[Path], root: Path, class_depth: int) -> bool:
    return any(_path_status_from_tokens(path, root, class_depth) in {"OK", "NG"} for path in image_paths)


def finalize_evaluation_outputs(
    predictions: pd.DataFrame,
    output_dir: Path,
    method_name: str,
    image_root: Path | None = None,
    class_depth: int = 1,
    histogram_bins: int = 30,
    montage_samples: int = 30,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _normalize_predictions(predictions, image_root, class_depth)
    predictions_path = output_dir / "predictions.csv"
    df.to_csv(predictions_path, index=False)

    _copy_classified_images(df, output_dir / "classified")
    _plot_prediction_counts(df, output_dir / "analysis", method_name)
    _plot_score_histograms(df, output_dir / "analysis", method_name, histogram_bins)
    _plot_top_scores(df, output_dir / "analysis", method_name)
    _export_top_score_montages(df, output_dir / "montage", montage_samples)

    summary_path = output_dir / "evaluation_summary.csv"
    summary = summarize_evaluation(df)
    summary.to_csv(summary_path, index=False)
    _plot_confusion_matrix(df, output_dir / "analysis", method_name)
    report_path = _write_evaluation_report(
        output_dir=output_dir,
        method_name=method_name,
        predictions_path=predictions_path,
        summary_path=summary_path,
        predictions=df,
    )
    return predictions_path, report_path


def summarize_evaluation(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary_row("overall", predictions)]
    class_column = _class_column(predictions)
    if class_column:
        for class_label, group in predictions.groupby(class_column):
            rows.append(_summary_row(str(class_label), group))
    return pd.DataFrame(rows)


def _normalize_predictions(
    predictions: pd.DataFrame,
    image_root: Path | None,
    class_depth: int,
) -> pd.DataFrame:
    df = predictions.copy()
    if "image_path" not in df:
        raise ValueError("predictions must contain image_path")
    df["source_image_path"] = df["image_path"].map(lambda value: str(Path(value).resolve()))
    df["pred_score"] = pd.to_numeric(df.get("pred_score"), errors="coerce")
    df["predicted_status"] = [
        _prediction_status(row)
        for row in df.to_dict(orient="records")
    ]
    df["prediction"] = df["predicted_status"]
    if "ground_truth_status" not in df:
        df["ground_truth_status"] = [
            _ground_truth_status(row, image_root, class_depth)
            for row in df.to_dict(orient="records")
        ]
    df["ground_truth_status"] = df["ground_truth_status"].fillna("UNKNOWN").map(_normalize_status)
    if "is_correct" not in df:
        df["is_correct"] = [
            pred == gt if gt in {"OK", "NG"} else np.nan
            for pred, gt in zip(df["predicted_status"], df["ground_truth_status"], strict=True)
        ]
    return df


def _prediction_status(row: dict) -> str:
    for key in ("prediction", "predicted_status", "pred_label"):
        if key in row and not pd.isna(row[key]):
            status = _normalize_status(row[key])
            if status != "UNKNOWN":
                return status
    score = row.get("pred_score")
    threshold = row.get("threshold")
    if score is not None and threshold is not None and not pd.isna(score) and not pd.isna(threshold):
        return "NG" if float(score) > float(threshold) else "OK"
    return "UNKNOWN"


def _ground_truth_status(row: dict, image_root: Path | None, class_depth: int) -> str:
    for key in ("ground_truth_status", "ground_truth", "gt_label"):
        if key in row and not pd.isna(row[key]):
            status = _normalize_status(row[key])
            if status != "UNKNOWN":
                return status
    label = row.get("label")
    if label is not None and not pd.isna(label):
        status = _normalize_status(label)
        if status != "UNKNOWN":
            return status
    if image_root is not None:
        return _path_status_from_tokens(Path(row["source_image_path"]), image_root, class_depth)
    return "UNKNOWN"


def _normalize_status(value: object) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().lower()
    if text in {"0", "false", "good", "normal", "ok", "pass"}:
        return "OK"
    if text in {"1", "true", "ng", "bad", "abnormal", "anomaly", "anomalous", "defect"}:
        return "NG"
    return "UNKNOWN"


def _path_status_from_tokens(path: Path, root: Path, class_depth: int) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return "UNKNOWN"
    tokens = {part.lower() for part in relative.parts[class_depth:-1]}
    if tokens & ANOMALY_TOKENS:
        return "NG"
    if tokens & GOOD_TOKENS:
        return "OK"
    return "UNKNOWN"


def _copy_classified_images(predictions: pd.DataFrame, output_dir: Path) -> None:
    for bucket in ("OK", "NG", "UNKNOWN"):
        (output_dir / bucket).mkdir(parents=True, exist_ok=True)
        (output_dir / f"{bucket}_overlays").mkdir(parents=True, exist_ok=True)

    for row in predictions.itertuples(index=False):
        bucket = str(row.predicted_status)
        source = Path(row.source_image_path)
        if source.exists():
            _copy_file(source, output_dir / bucket)
        overlay_value = str(getattr(row, "overlay_path", "")).strip()
        if overlay_value:
            overlay_path = Path(overlay_value)
            if overlay_path.is_file():
                _copy_file(overlay_path, output_dir / f"{bucket}_overlays")


def _copy_file(source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name
    if target.exists():
        target = output_dir / f"{source.stem}_{abs(hash(source))}{source.suffix}"
    shutil.copy2(source, target)


def _plot_prediction_counts(predictions: pd.DataFrame, output_dir: Path, method_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = predictions["predicted_status"].value_counts().reindex(["OK", "NG", "UNKNOWN"], fill_value=0)
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#3b7d4f", "#b73737", "#777777"])
    plt.title(f"{method_name} prediction counts")
    plt.xlabel("prediction")
    plt.ylabel("count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_counts.png", dpi=160)
    plt.close()


def _plot_score_histograms(
    predictions: pd.DataFrame,
    output_dir: Path,
    method_name: str,
    bins: int,
) -> None:
    scores = pd.to_numeric(predictions["pred_score"], errors="coerce")
    if scores.dropna().empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_grouped_histogram(
        predictions,
        group_column="predicted_status",
        title=f"{method_name} scores by prediction",
        xlabel="pred_score",
        output_path=output_dir / "score_histogram_by_prediction.png",
        bins=bins,
    )
    labeled = predictions[predictions["ground_truth_status"].isin(["OK", "NG"])]
    if not labeled.empty:
        _plot_grouped_histogram(
            labeled,
            group_column="ground_truth_status",
            title=f"{method_name} scores by ground truth",
            xlabel="pred_score",
            output_path=output_dir / "score_histogram_by_ground_truth.png",
            bins=bins,
        )


def _plot_grouped_histogram(
    predictions: pd.DataFrame,
    group_column: str,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, group in predictions.groupby(group_column):
        pd.to_numeric(group["pred_score"], errors="coerce").dropna().hist(
            bins=bins,
            alpha=0.55,
            label=str(label),
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_top_scores(predictions: pd.DataFrame, output_dir: Path, method_name: str) -> None:
    scored = predictions.copy()
    scored["pred_score"] = pd.to_numeric(scored["pred_score"], errors="coerce")
    scored = scored.dropna(subset=["pred_score"]).sort_values("pred_score", ascending=False)
    if scored.empty:
        return
    top = scored.head(min(50, len(scored)))
    plt.figure(figsize=(10, max(4, min(12, len(top) * 0.24))))
    plt.barh(
        [Path(path).name for path in top["source_image_path"]][::-1],
        top["pred_score"].to_numpy()[::-1],
        color="#b73737",
    )
    plt.title(f"Top {method_name} anomaly scores")
    plt.xlabel("pred_score")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "top_anomaly_scores.png", dpi=160)
    plt.close()


def _export_top_score_montages(predictions: pd.DataFrame, output_dir: Path, samples: int) -> None:
    scored = predictions.copy()
    scored["pred_score"] = pd.to_numeric(scored["pred_score"], errors="coerce")
    scored = scored.dropna(subset=["pred_score"]).sort_values("pred_score", ascending=False)
    if scored.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    top = scored.head(samples)
    _save_montage(top, "source_image_path", output_dir / "top_anomaly_scores.jpg")
    if "overlay_path" in top:
        overlays = top[top["overlay_path"].astype(str).str.len() > 0]
        _save_montage(overlays, "overlay_path", output_dir / "top_anomaly_overlays.jpg")


def _save_montage(df: pd.DataFrame, path_column: str, output_path: Path) -> None:
    if df.empty:
        return
    tile_size = (180, 120)
    columns = 5
    rows = max(1, (len(df) + columns - 1) // columns)
    montage = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(montage)
    for index, row in enumerate(df.itertuples(index=False)):
        image_path = Path(str(getattr(row, path_column)))
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
            draw.rectangle([tile_x, tile_y, tile_x + tile_size[0] - 1, tile_y + tile_size[1] - 1], outline=(220, 0, 0), width=2)
            score = getattr(row, "pred_score", None)
            if score is not None and not pd.isna(score):
                draw.text((tile_x + 4, tile_y + 4), f"{float(score):.3f}", fill=(0, 0, 0))
    montage.save(output_path)


def _plot_confusion_matrix(predictions: pd.DataFrame, output_dir: Path, method_name: str) -> None:
    labeled = predictions[predictions["ground_truth_status"].isin(["OK", "NG"])]
    if labeled.empty:
        return
    y_true = labeled["ground_truth_status"]
    y_pred = labeled["predicted_status"].where(labeled["predicted_status"].isin(["OK", "NG"]), "UNKNOWN")
    labels = ["OK", "NG", "UNKNOWN"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{method_name} confusion matrix")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("predicted")
    plt.ylabel("ground truth")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x, y, str(matrix[y, x]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def _summary_row(name: str, group: pd.DataFrame) -> dict[str, object]:
    scores = pd.to_numeric(group["pred_score"], errors="coerce")
    row: dict[str, object] = {
        "group": name,
        "count": len(group),
        "ok_count": int((group["predicted_status"] == "OK").sum()),
        "ng_count": int((group["predicted_status"] == "NG").sum()),
        "unknown_count": int((group["predicted_status"] == "UNKNOWN").sum()),
        "score_mean": float(scores.mean()) if not scores.dropna().empty else np.nan,
        "score_p95": float(scores.quantile(0.95)) if not scores.dropna().empty else np.nan,
    }
    labeled = group[group["ground_truth_status"].isin(["OK", "NG"])]
    if not labeled.empty:
        y_true = (labeled["ground_truth_status"] == "NG").astype(int)
        y_pred = (labeled["predicted_status"] == "NG").astype(int)
        row.update(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
        if len(set(y_true)) > 1 and not pd.to_numeric(labeled["pred_score"], errors="coerce").dropna().empty:
            row["roc_auc"] = float(roc_auc_score(y_true, pd.to_numeric(labeled["pred_score"], errors="coerce")))
    return row


def _class_column(predictions: pd.DataFrame) -> str | None:
    for column in ("class_label", "class", "connector_class"):
        if column in predictions:
            return column
    return None


def _write_evaluation_report(
    output_dir: Path,
    method_name: str,
    predictions_path: Path,
    summary_path: Path,
    predictions: pd.DataFrame,
) -> Path:
    counts = predictions["predicted_status"].value_counts().reindex(["OK", "NG", "UNKNOWN"], fill_value=0)
    labeled_count = int(predictions["ground_truth_status"].isin(["OK", "NG"]).sum())
    lines = [
        f"# {method_name} Evaluation Report",
        "",
        f"- Predictions: `{predictions_path}`",
        f"- Summary: `{summary_path}`",
        f"- Total images: {len(predictions)}",
        f"- Labeled images: {labeled_count}",
        f"- Predicted OK: {int(counts['OK'])}",
        f"- Predicted NG: {int(counts['NG'])}",
        f"- Predicted UNKNOWN: {int(counts['UNKNOWN'])}",
        "",
        "## Artifacts",
        "",
        "- `classified/OK`: original images predicted OK",
        "- `classified/NG`: original images predicted NG",
        "- `classified/UNKNOWN`: images without a usable prediction",
        "- `analysis/prediction_counts.png`: OK/NG/UNKNOWN count chart",
        "- `analysis/score_histogram_by_prediction.png`: score distribution by prediction",
        "- `analysis/top_anomaly_scores.png`: highest anomaly score chart",
        "- `analysis/confusion_matrix.png`: confusion matrix when OK/NG labels are available",
        "- `montage/top_anomaly_scores.jpg`: highest-score image montage",
    ]
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
