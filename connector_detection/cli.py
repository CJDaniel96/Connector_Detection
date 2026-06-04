from __future__ import annotations

from pathlib import Path

import typer

from connector_detection.clustering import assign_existing_embeddings, fit_clusters
from connector_detection.config import load_config
from connector_detection.features import extract_embeddings
from connector_detection.review import export_review_samples
from connector_detection.visualize import plot_umap

app = typer.Typer(help="Connector anomaly research pipeline.")


@app.command()
def extract(config: Path, device: str | None = None) -> None:
    cfg = load_config(config)
    embedding_path, manifest_path = extract_embeddings(
        image_dir=cfg.image_dir,
        output_dir=cfg.output_dir,
        model_name=cfg.dinov2_model,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        device=device,
    )
    typer.echo(f"Saved {embedding_path}")
    typer.echo(f"Saved {manifest_path}")


@app.command()
def cluster(config: Path) -> None:
    cfg = load_config(config)
    clusters_path, model_path = fit_clusters(
        embeddings_path=cfg.output_dir / "embeddings.npy",
        manifest_path=cfg.output_dir / "manifest.csv",
        output_dir=cfg.output_dir,
        pca_components=cfg.pca_components,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        unknown_distance_quantile=cfg.unknown_distance_quantile,
        random_state=cfg.random_state,
    )
    typer.echo(f"Saved {clusters_path}")
    typer.echo(f"Saved {model_path}")


@app.command()
def review(config: Path) -> None:
    cfg = load_config(config)
    summary_path = export_review_samples(
        clusters_csv=cfg.output_dir / "clusters.csv",
        output_dir=cfg.output_dir / "review",
        samples_per_cluster=cfg.review_samples_per_cluster,
        random_state=cfg.random_state,
    )
    typer.echo(f"Saved {summary_path}")


@app.command()
def umap(config: Path) -> None:
    cfg = load_config(config)
    output_path = plot_umap(
        pca_embeddings_path=cfg.output_dir / "embeddings_pca.npy",
        clusters_csv=cfg.output_dir / "clusters.csv",
        output_path=cfg.output_dir / "umap_clusters.png",
        random_state=cfg.random_state,
    )
    typer.echo(f"Saved {output_path}")


@app.command()
def assign(
    embeddings: Path,
    model: Path,
    output: Path = Path("outputs/assignments.csv"),
) -> None:
    output_path = assign_existing_embeddings(embeddings, model, output)
    typer.echo(f"Saved {output_path}")


@app.command()
def run(config: Path, device: str | None = None) -> None:
    extract(config, device)
    cluster(config)
    review(config)
    umap(config)
