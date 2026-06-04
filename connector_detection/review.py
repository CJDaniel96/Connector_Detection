from __future__ import annotations

from pathlib import Path

import pandas as pd

from connector_detection.images import copy_review_images


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
    return summary_path
