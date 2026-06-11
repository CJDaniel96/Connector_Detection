from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from connector_detection.images import list_images, resize_with_padding

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "connector_detection_matplotlib"),
)
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FoundationClusterConfig:
    model_kind: str = "dinov2"
    model_name: str | None = None
    image_size: int = 224
    batch_size: int = 16
    reducer: str = "pca"
    reduced_dim: int = 50
    n_clusters: int = 8
    umap_neighbors: int = 15
    review_samples: int = 25
    random_state: int = 42
    normalize_embeddings: bool = True


def run_foundation_clustering(
    image_dir: Path,
    output_dir: Path,
    config: FoundationClusterConfig,
    device: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    image_dir = image_dir.resolve()
    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")
    if config.n_clusters > len(image_paths):
        raise ValueError(
            f"n_clusters={config.n_clusters} is larger than image count={len(image_paths)}"
        )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(image_paths, image_dir, output_dir)
    embeddings = _extract_foundation_embeddings(image_paths, config, device=device)
    if config.normalize_embeddings:
        embeddings = _l2_normalize(embeddings)
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    reduced_embeddings, reducer_model, feature_scaler = _reduce_embeddings(embeddings, config)
    reduced_path = output_dir / "embeddings_reduced.npy"
    np.save(reduced_path, reduced_embeddings)

    clusters, kmeans = _fit_kmeans(reduced_embeddings, config)
    clusters_path = _write_clusters(image_paths, image_dir, clusters, output_dir)
    umap_path = _plot_umap(
        embeddings=reduced_embeddings,
        clusters=clusters["cluster"],
        output_path=output_dir / "umap_clusters.png",
        random_state=config.random_state,
        n_neighbors=config.umap_neighbors,
    )
    review_summary_path = _export_cluster_review(
        clusters_csv=clusters_path,
        output_dir=output_dir / "review",
        review_samples=config.review_samples,
    )
    labels_template_path = _write_label_template(clusters_path, output_dir)
    model_path = _write_model(
        output_dir=output_dir,
        config=config,
        reducer_model=reducer_model,
        feature_scaler=feature_scaler,
        kmeans=kmeans,
        image_dir=image_dir,
        manifest_path=manifest_path,
        embeddings_path=embeddings_path,
        reduced_path=reduced_path,
        clusters_path=clusters_path,
    )
    _write_report(
        output_dir=output_dir,
        config=config,
        image_count=len(image_paths),
        clusters_path=clusters_path,
        umap_path=umap_path,
        review_summary_path=review_summary_path,
        labels_template_path=labels_template_path,
        model_path=model_path,
    )
    return clusters_path, umap_path, review_summary_path, labels_template_path


def apply_foundation_cluster_labels(
    clusters_csv: Path,
    labels_csv: Path,
    output_path: Path,
) -> Path:
    clusters = pd.read_csv(clusters_csv)
    labels = pd.read_csv(labels_csv)
    required = {"cluster", "cluster_name", "merge_to"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"labels_csv is missing columns: {', '.join(sorted(missing))}")

    label_rows = {}
    for _, row in labels.iterrows():
        cluster = _parse_cluster_id(row["cluster"])
        merge_to = row.get("merge_to", "")
        final_cluster = _parse_cluster_id(merge_to) if pd.notna(merge_to) and str(merge_to).strip() else cluster
        label_rows[cluster] = {
            "cluster_name": "" if pd.isna(row.get("cluster_name", "")) else row["cluster_name"],
            "final_cluster": final_cluster,
        }

    clusters["review_cluster"] = clusters["cluster"].astype(int)
    clusters["final_cluster"] = clusters["review_cluster"].map(
        lambda value: label_rows.get(int(value), {}).get("final_cluster", int(value))
    )
    clusters["cluster_name"] = clusters["review_cluster"].map(
        lambda value: label_rows.get(int(value), {}).get("cluster_name", "")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(output_path, index=False)
    return output_path


def _parse_cluster_id(value: object) -> int:
    return int(float(str(value).strip()))


def _write_manifest(image_paths: list[Path], image_dir: Path, output_dir: Path) -> Path:
    rows = [
        {
            "image_path": str(path),
            "relative_path": str(path.relative_to(image_dir)),
        }
        for path in image_paths
    ]
    manifest_path = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _extract_foundation_embeddings(
    image_paths: list[Path],
    config: FoundationClusterConfig,
    device: str | None,
) -> np.ndarray:
    kind = config.model_kind.lower()
    if kind in {"dinov2", "vit"}:
        return _extract_hf_vision_embeddings(image_paths, config, device=device)
    if kind == "clip":
        return _extract_clip_embeddings(image_paths, config, device=device)
    if kind == "resnet":
        return _extract_resnet_embeddings(image_paths, config, device=device)
    raise ValueError("model_kind must be one of: dinov2, clip, vit, resnet")


def _extract_hf_vision_embeddings(
    image_paths: list[Path],
    config: FoundationClusterConfig,
    device: str | None,
) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    model_name = config.model_name or _default_model_name(config.model_kind)
    torch_device = _resolve_device(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(torch_device)
    model.eval()

    vectors: list[np.ndarray] = []
    for batch_paths in _batched(image_paths, config.batch_size):
        images = [_load_rgb(path, config.image_size) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            if getattr(outputs, "pooler_output", None) is not None:
                batch_vectors = outputs.pooler_output
            else:
                batch_vectors = outputs.last_hidden_state[:, 0]
        vectors.append(batch_vectors.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def _extract_clip_embeddings(
    image_paths: list[Path],
    config: FoundationClusterConfig,
    device: str | None,
) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_name = config.model_name or _default_model_name("clip")
    torch_device = _resolve_device(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(torch_device)
    model.eval()

    vectors: list[np.ndarray] = []
    for batch_paths in _batched(image_paths, config.batch_size):
        images = [_load_rgb(path, config.image_size) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}
        with torch.inference_mode():
            batch_vectors = model.get_image_features(**inputs)
        vectors.append(batch_vectors.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def _extract_resnet_embeddings(
    image_paths: list[Path],
    config: FoundationClusterConfig,
    device: str | None,
) -> np.ndarray:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    torch_device = _resolve_device(device)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(torch_device)
    model.eval()
    transform = weights.transforms()

    vectors: list[np.ndarray] = []
    for batch_paths in _batched(image_paths, config.batch_size):
        tensors = [transform(_load_rgb(path, config.image_size)) for path in batch_paths]
        batch = torch.stack(tensors).to(torch_device)
        with torch.inference_mode():
            batch_vectors = model(batch)
        vectors.append(batch_vectors.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def _default_model_name(model_kind: str) -> str:
    kind = model_kind.lower()
    if kind == "clip":
        return "openai/clip-vit-base-patch32"
    if kind == "vit":
        return "google/vit-base-patch16-224-in21k"
    if kind == "dinov2":
        return "facebook/dinov2-small"
    if kind == "resnet":
        return "torchvision/resnet50"
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def _resolve_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_rgb(path: Path, image_size: int) -> Image.Image:
    with Image.open(path) as image:
        return resize_with_padding(ImageOps.exif_transpose(image), image_size)


def _batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def _reduce_embeddings(
    embeddings: np.ndarray,
    config: FoundationClusterConfig,
) -> tuple[np.ndarray, object, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    n_components = min(config.reduced_dim, scaled.shape[0], scaled.shape[1])
    if n_components < 1:
        raise ValueError("Cannot reduce an empty embedding matrix")

    reducer_name = config.reducer.lower()
    if reducer_name == "pca":
        reducer = PCA(n_components=n_components, random_state=config.random_state)
    elif reducer_name == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=_safe_umap_neighbors(scaled.shape[0], config.umap_neighbors),
            random_state=config.random_state,
        )
    else:
        raise ValueError("reducer must be either pca or umap")
    reduced = reducer.fit_transform(scaled).astype(np.float32)
    return reduced, reducer, scaler


def _fit_kmeans(
    reduced_embeddings: np.ndarray,
    config: FoundationClusterConfig,
) -> tuple[pd.DataFrame, KMeans]:
    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init="auto",
    )
    cluster_ids = kmeans.fit_predict(reduced_embeddings)
    distances = np.linalg.norm(reduced_embeddings - kmeans.cluster_centers_[cluster_ids], axis=1)
    rows = pd.DataFrame(
        {
            "cluster": cluster_ids.astype(int),
            "distance_to_centroid": distances.astype(float),
        }
    )
    rows["representative_rank"] = (
        rows.groupby("cluster")["distance_to_centroid"].rank(method="first", ascending=True).astype(int)
    )
    return rows, kmeans


def _write_clusters(
    image_paths: list[Path],
    image_dir: Path,
    clusters: pd.DataFrame,
    output_dir: Path,
) -> Path:
    rows = clusters.copy()
    rows.insert(0, "image_path", [str(path) for path in image_paths])
    rows.insert(1, "relative_path", [str(path.relative_to(image_dir)) for path in image_paths])
    clusters_path = output_dir / "clusters.csv"
    rows.to_csv(clusters_path, index=False)
    return clusters_path


def _plot_umap(
    embeddings: np.ndarray,
    clusters: pd.Series,
    output_path: Path,
    random_state: int,
    n_neighbors: int,
) -> Path:
    import umap

    if embeddings.shape[0] < 3:
        points = _fallback_2d_points(embeddings)
    else:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=_safe_umap_neighbors(embeddings.shape[0], n_neighbors),
            random_state=random_state,
        )
        points = reducer.fit_transform(embeddings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=clusters,
        cmap="tab20",
        s=18,
        alpha=0.85,
    )
    plt.colorbar(scatter, label="cluster")
    plt.title("Foundation Embedding K-Means Clusters")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _fallback_2d_points(embeddings: np.ndarray) -> np.ndarray:
    points = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
    if embeddings.shape[1] > 0:
        points[:, 0] = embeddings[:, 0]
    if embeddings.shape[1] > 1:
        points[:, 1] = embeddings[:, 1]
    return points


def _safe_umap_neighbors(sample_count: int, requested: int) -> int:
    if sample_count <= 2:
        return 2
    return max(2, min(requested, sample_count - 1))


def _export_cluster_review(
    clusters_csv: Path,
    output_dir: Path,
    review_samples: int,
) -> Path:
    clusters = pd.read_csv(clusters_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    montage_dir = output_dir / "montage"
    summary_rows = []
    for cluster, group in clusters.groupby("cluster"):
        group = group.sort_values("representative_rank")
        sample_count = min(review_samples, len(group))
        samples = group.head(sample_count)
        cluster_dir = output_dir / f"cluster_{int(cluster):03d}"
        _copy_review_images(samples, cluster_dir)
        montage_path = montage_dir / f"cluster_{int(cluster):03d}.jpg"
        _make_montage(
            [Path(path) for path in samples["image_path"]],
            montage_path,
            labels=[f"rank {rank}" for rank in samples["representative_rank"]],
        )
        summary_rows.append(
            {
                "cluster": int(cluster),
                "count": len(group),
                "review_count": sample_count,
                "mean_distance_to_centroid": float(group["distance_to_centroid"].mean()),
                "median_distance_to_centroid": float(group["distance_to_centroid"].median()),
                "review_dir": str(cluster_dir),
                "montage_path": str(montage_path),
            }
        )
    summary_path = output_dir / "review_summary.csv"
    pd.DataFrame(summary_rows).sort_values("cluster").to_csv(summary_path, index=False)
    return summary_path


def _copy_review_images(samples: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in samples.iterrows():
        source = Path(row["image_path"])
        target = output_dir / f"rank_{int(row['representative_rank']):03d}_{source.name}"
        if target.exists():
            target = output_dir / f"rank_{int(row['representative_rank']):03d}_{abs(hash(source))}_{source.name}"
        with Image.open(source) as image:
            ImageOps.exif_transpose(image).save(target)


def _make_montage(
    image_paths: list[Path],
    output_path: Path,
    labels: list[str] | None = None,
    tile_size: tuple[int, int] = (180, 120),
    columns: int = 5,
) -> None:
    if not image_paths:
        return
    rows = math.ceil(len(image_paths) / columns)
    montage = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    draw = ImageDraw.Draw(montage)
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            image.thumbnail((tile_size[0], tile_size[1] - 18), Image.Resampling.BICUBIC)
            tile_x = (index % columns) * tile_size[0]
            tile_y = (index // columns) * tile_size[1]
            x = tile_x + (tile_size[0] - image.width) // 2
            y = tile_y + 4
            montage.paste(image, (x, y))
            draw.rectangle(
                [tile_x, tile_y, tile_x + tile_size[0] - 1, tile_y + tile_size[1] - 1],
                outline=(220, 220, 220),
            )
            if labels:
                draw.text((tile_x + 6, tile_y + tile_size[1] - 16), labels[index], fill=(30, 30, 30))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)


def _write_label_template(clusters_path: Path, output_dir: Path) -> Path:
    clusters = pd.read_csv(clusters_path)
    rows = []
    for cluster, group in clusters.groupby("cluster"):
        rows.append(
            {
                "cluster": int(cluster),
                "count": len(group),
                "cluster_name": "",
                "merge_to": "",
                "split_note": "",
                "decision": "keep",
            }
        )
    template_path = output_dir / "cluster_labels_template.csv"
    pd.DataFrame(rows).sort_values("cluster").to_csv(template_path, index=False)
    return template_path


def _write_model(
    output_dir: Path,
    config: FoundationClusterConfig,
    reducer_model: object,
    feature_scaler: StandardScaler,
    kmeans: KMeans,
    image_dir: Path,
    manifest_path: Path,
    embeddings_path: Path,
    reduced_path: Path,
    clusters_path: Path,
) -> Path:
    model_path = output_dir / "foundation_cluster_model.joblib"
    joblib.dump(
        {
            "config": asdict(config),
            "feature_scaler": feature_scaler,
            "reducer": reducer_model,
            "kmeans": kmeans,
            "image_dir": image_dir,
            "manifest_path": manifest_path,
            "embeddings_path": embeddings_path,
            "reduced_embeddings_path": reduced_path,
            "clusters_path": clusters_path,
        },
        model_path,
    )
    with (output_dir / "foundation_cluster_config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)
    return model_path


def _write_report(
    output_dir: Path,
    config: FoundationClusterConfig,
    image_count: int,
    clusters_path: Path,
    umap_path: Path,
    review_summary_path: Path,
    labels_template_path: Path,
    model_path: Path,
) -> None:
    clusters = pd.read_csv(clusters_path)
    summary = clusters.groupby("cluster").size().reset_index(name="count")
    lines = [
        "# Foundation Clustering Report",
        "",
        f"- image_count: {image_count}",
        f"- model_kind: {config.model_kind}",
        f"- model_name: {config.model_name or _default_model_name(config.model_kind)}",
        f"- reducer: {config.reducer}",
        f"- reduced_dim: {config.reduced_dim}",
        f"- n_clusters: {config.n_clusters}",
        f"- clusters_csv: {clusters_path}",
        f"- umap_plot: {umap_path}",
        f"- review_summary: {review_summary_path}",
        f"- label_template: {labels_template_path}",
        f"- model: {model_path}",
        "",
        "## Cluster Counts",
        "",
        _dataframe_to_markdown(summary),
    ]
    (output_dir / "foundation_clustering_report.md").write_text("\n".join(lines), encoding="utf-8")


def _dataframe_to_markdown(data: pd.DataFrame) -> str:
    if data.empty:
        return "_No rows._"
    columns = list(data.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)
