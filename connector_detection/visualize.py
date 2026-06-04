from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_umap(
    pca_embeddings_path: Path,
    clusters_csv: Path,
    output_path: Path,
    random_state: int,
) -> Path:
    import umap

    embeddings = np.load(pca_embeddings_path)
    clusters = pd.read_csv(clusters_csv)
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    points = reducer.fit_transform(embeddings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=clusters["cluster"],
        cmap="tab20",
        s=8,
        alpha=0.8,
    )
    plt.colorbar(scatter, label="cluster")
    plt.title("Connector crop UMAP by HDBSCAN cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
