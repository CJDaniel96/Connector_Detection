from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class ClusterAssignmentModel:
    pca: PCA
    centroids: dict[int, np.ndarray]
    knn: KNeighborsClassifier
    unknown_threshold: float

    def predict(self, embeddings: np.ndarray) -> pd.DataFrame:
        reduced = self.pca.transform(embeddings)
        labels = self.knn.predict(reduced)
        distances = []
        final_labels = []
        for vector, label in zip(reduced, labels, strict=True):
            centroid = self.centroids[int(label)]
            distance = float(np.linalg.norm(vector - centroid))
            distances.append(distance)
            final_labels.append(-1 if distance > self.unknown_threshold else int(label))
        return pd.DataFrame({"cluster": final_labels, "centroid_distance": distances})


def fit_clusters(
    embeddings_path: Path,
    manifest_path: Path,
    output_dir: Path,
    pca_components: int,
    min_cluster_size: int,
    min_samples: int | None,
    unknown_distance_quantile: float,
    random_state: int,
) -> tuple[Path, Path]:
    import hdbscan

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(embeddings_path)
    manifest = pd.read_csv(manifest_path)

    n_components = min(pca_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    cluster_df = manifest.copy()
    cluster_df["cluster"] = labels
    cluster_df["hdbscan_probability"] = clusterer.probabilities_
    cluster_df.to_csv(output_dir / "clusters.csv", index=False)
    np.save(output_dir / "embeddings_pca.npy", reduced)

    valid_mask = labels >= 0
    if not np.any(valid_mask):
        raise ValueError("HDBSCAN produced no non-noise clusters. Lower min_cluster_size.")

    centroids = {
        int(label): reduced[labels == label].mean(axis=0)
        for label in sorted(set(labels))
        if label >= 0
    }
    knn = KNeighborsClassifier(n_neighbors=min(5, int(valid_mask.sum())))
    knn.fit(reduced[valid_mask], labels[valid_mask])

    direct_distances = np.array(
        [
            np.linalg.norm(vector - centroids[int(label)])
            for vector, label in zip(reduced[valid_mask], labels[valid_mask], strict=True)
        ]
    )
    unknown_threshold = float(np.quantile(direct_distances, unknown_distance_quantile))

    assignment_model = ClusterAssignmentModel(
        pca=pca,
        centroids=centroids,
        knn=knn,
        unknown_threshold=unknown_threshold,
    )
    model_path = output_dir / "cluster_assignment.joblib"
    joblib.dump(
        {
            "assignment_model": assignment_model,
            "hdbscan_clusterer": clusterer,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        },
        model_path,
    )
    return output_dir / "clusters.csv", model_path


def assign_existing_embeddings(
    embeddings_path: Path,
    model_path: Path,
    output_path: Path,
) -> Path:
    payload = joblib.load(model_path)
    model: ClusterAssignmentModel = payload["assignment_model"]
    embeddings = np.load(embeddings_path)
    predictions = model.predict(embeddings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return output_path
