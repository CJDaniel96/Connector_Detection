from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PipelineConfig:
    image_dir: Path
    output_dir: Path
    image_size: int = 224
    dinov2_model: str = "facebook/dinov2-small"
    batch_size: int = 16
    pca_components: int = 50
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int | None = None
    review_samples_per_cluster: int = 30
    unknown_distance_quantile: float = 0.995
    random_state: int = 42


def load_config(path: Path) -> PipelineConfig:
    data = tomllib.loads(path.read_text())
    section = data.get("pipeline", data)
    return PipelineConfig(
        image_dir=Path(section["image_dir"]).expanduser(),
        output_dir=Path(section.get("output_dir", "outputs")).expanduser(),
        image_size=int(section.get("image_size", 224)),
        dinov2_model=str(section.get("dinov2_model", "facebook/dinov2-small")),
        batch_size=int(section.get("batch_size", 16)),
        pca_components=int(section.get("pca_components", 50)),
        hdbscan_min_cluster_size=int(section.get("hdbscan_min_cluster_size", 15)),
        hdbscan_min_samples=(
            None
            if section.get("hdbscan_min_samples") in (None, "null")
            else int(section["hdbscan_min_samples"])
        ),
        review_samples_per_cluster=int(section.get("review_samples_per_cluster", 30)),
        unknown_distance_quantile=float(section.get("unknown_distance_quantile", 0.995)),
        random_state=int(section.get("random_state", 42)),
    )
