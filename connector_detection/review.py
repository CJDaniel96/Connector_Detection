from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from connector_detection.images import copy_review_images


def _make_montage(
    image_paths: list[Path],
    output_path: Path,
    tile_size: tuple[int, int] = (160, 80),
    columns: int = 5,
) -> None:
    if not image_paths:
        return

    rows = (len(image_paths) + columns - 1) // columns
    montage = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(montage)
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            image.thumbnail(tile_size, Image.Resampling.BICUBIC)
            x = (index % columns) * tile_size[0] + (tile_size[0] - image.width) // 2
            y = (index // columns) * tile_size[1] + (tile_size[1] - image.height) // 2
            montage.paste(image, (x, y))
            draw.rectangle(
                [
                    (index % columns) * tile_size[0],
                    (index // columns) * tile_size[1],
                    (index % columns + 1) * tile_size[0] - 1,
                    (index // columns + 1) * tile_size[1] - 1,
                ],
                outline=(220, 220, 220),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)


def _cluster_feature_summary(clusters: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "aspect_ratio",
        "edge_density",
        "bright_ratio",
        "mean_intensity",
        "std_intensity",
        "peak_count",
        "peak_spacing_mean",
        "peak_spacing_std",
    ]
    available = [column for column in feature_columns if column in clusters.columns]
    if not available:
        return pd.DataFrame()

    rows = []
    for cluster, group in clusters.groupby("cluster"):
        row = {"cluster": cluster, "count": len(group)}
        for column in available:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
            row[f"{column}_median"] = float(group[column].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster")


def export_review_samples(
    clusters_csv: Path,
    output_dir: Path,
    samples_per_cluster: int,
    random_state: int,
) -> Path:
    clusters = pd.read_csv(clusters_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for cluster, group in clusters.groupby("cluster"):
        group = group.sort_values("hdbscan_probability", ascending=False)
        sample_count = min(samples_per_cluster, len(group))
        samples = group.head(sample_count)
        cluster_dir = output_dir / f"cluster_{cluster}"
        copy_review_images([Path(p) for p in samples["image_path"]], cluster_dir)
        _make_montage(
            [Path(p) for p in samples["image_path"]],
            output_dir / "montage" / f"cluster_{cluster}.jpg",
        )
        summary_rows.append(
            {
                "cluster": cluster,
                "count": len(group),
                "review_dir": str(cluster_dir),
                "mean_probability": float(group["hdbscan_probability"].mean()),
            }
        )

    summary_path = output_dir / "review_summary.csv"
    pd.DataFrame(summary_rows).sort_values("cluster").to_csv(summary_path, index=False)
    feature_summary = _cluster_feature_summary(clusters)
    if not feature_summary.empty:
        feature_summary.to_csv(output_dir / "cluster_feature_summary.csv", index=False)
    return summary_path
