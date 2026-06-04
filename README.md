# Connector Detection

AI-based connector inspection research pipeline for connector crops, pin-band
grouping, and cluster-level anomaly workflows.

## Goal

The first implementation focuses on the clustering backbone:

1. Use your existing YOLO model to crop connector components.
2. Resize and pad connector crops to a stable square input.
3. Extract DINOv2 embeddings.
4. Reduce embeddings with PCA.
5. Cluster connector types with HDBSCAN.
6. Export representative samples for manual review.
7. Plot UMAP for cluster quality inspection.
8. Save a KNN / nearest-centroid assignment model for new samples.

After the cluster groups are stable, each final cluster can own its own pin-band
YOLO model or crop config, PatchCore feature bank, anomaly threshold, and
profile-check rules.

## Setup

```bash
uv sync
```

## Run the clustering pipeline

Edit `configs/pipeline.example.toml`, then run:

```bash
uv run connector-detection extract configs/pipeline.example.toml
uv run connector-detection cluster configs/pipeline.example.toml
uv run connector-detection review configs/pipeline.example.toml
uv run connector-detection umap configs/pipeline.example.toml
```

Or run all current stages:

```bash
uv run connector-detection run configs/pipeline.example.toml
```

Main outputs:

- `embeddings.npy`: DINOv2 connector embeddings
- `manifest.csv`: image path index aligned with embeddings
- `embeddings_pca.npy`: PCA-reduced feature vectors
- `clusters.csv`: HDBSCAN cluster result and confidence
- `review/`: sampled images grouped by cluster for human review
- `umap_clusters.png`: 2D UMAP visualization
- `cluster_assignment.joblib`: PCA + KNN + centroid unknown-threshold model

## Suggested R&D milestones

1. **Data contract**
   - Freeze folder naming for connector crops and pin-band crops.
   - Keep traceability from source image -> connector crop -> pin-band crop.

2. **Connector clustering**
   - Run DINOv2 + PCA + HDBSCAN.
   - Review cluster samples and merge/split connector families manually.
   - Tune `hdbscan_min_cluster_size`, `hdbscan_min_samples`, and PCA dimensions.

3. **Cluster assignment**
   - Use the saved KNN / centroid model for new connector crops.
   - Treat samples beyond the learned centroid distance threshold as `unknown`.
   - Periodically review unknowns and promote them to new profiles when needed.

4. **Pin-band profile per cluster**
   - For each approved connector profile, define either a pin-band YOLO model or
     deterministic crop config.
   - Record expected pin count, approximate pitch, orientation, and allowed
     geometric tolerance.

5. **Anomaly pipeline per cluster**
   - Build PatchCore feature banks from good pin-band samples.
   - Tune cluster-level thresholds on validation images.
   - Add rule checks for missing pin, shifted pin, occlusion, foreign material,
     and dirty regions.

6. **Production loop**
   - Log every prediction with source path, assigned cluster, distance, anomaly
     score, rule failures, and reviewed label.
   - Retrain clustering and anomaly banks when enough reviewed unknown or false
     reject samples accumulate.
