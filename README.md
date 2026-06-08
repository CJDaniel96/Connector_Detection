# Connector Detection

AI-based connector inspection research pipeline for connector crops, pin-band
grouping, and cluster-level anomaly workflows.

## Goal

The implementation focuses on the clustering backbone:

1. Use your existing YOLO model to crop connector components.
2. Resize and pad connector crops to a stable square input.
3. Extract DINOv2 embeddings.
4. Compute structural pin-band features: aspect ratio, edge density,
   projection profile, peak count, peak spacing, and bright ratio.
5. Standardize structural features, multiply by `structural_weight`, and
   concatenate them with DINOv2 embeddings.
6. Reduce combined features with PCA.
7. Cluster connector or pin-band types with HDBSCAN.
8. Export representative samples, montage images, and cluster feature
   statistics for manual review.
9. Plot UMAP for cluster quality inspection.
10. Save a KNN / nearest-centroid assignment model for new samples.

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

## Crop pin bands from PASCAL VOC XML

If you already have connector images and pin-band PASCAL VOC XML labels, crop
pin bands and rotate every crop into the same `up` orientation with:

```bash
uv run connector-detection crop-pin-bands \
  --xml-dir data/pin_band_xml \
  --image-dir data/connector_images \
  --output-dir outputs/pin_band_crops \
  --label pin_band \
  --padding 2
```

Outputs:

- normalized pin-band crops directly under `outputs/pin_band_crops`
- `outputs/pin_band_crops/pin_band_crops_manifest.csv`

Orientation rule:

- Horizontal bbox, `bbox_width >= bbox_height`: center above image center -> `up`,
  otherwise `down`.
- Vertical bbox, `bbox_width < bbox_height`: center left of image center -> `left`,
  otherwise `right`.

Normalization rule:

- `up`: keep original crop.
- `down`: rotate 180 degrees.
- `left`: rotate 90 degrees clockwise.
- `right`: rotate 90 degrees counter-clockwise.

## Fit nearest-centroid groups from manual labels

Put manually classified pin-band images into folders where each folder name is
the label:

```text
data/labeled_pin_bands/
  20pin/
    a.png
    b.png
  24pin/
    c.png
    d.png
```

Then fit a nearest-centroid model using the same DINOv2 + structural feature
input:

```bash
uv run connector-detection fit-centroids \
  configs/pipeline.example.toml \
  --labeled-image-dir data/labeled_pin_bands \
  --output-dir outputs/nearest_centroid_pin_groups
```

Outputs:

- `embeddings.npy`: final DINOv2 + weighted structural features
- `nearest_centroid_model.joblib`: label centroids and unknown thresholds
- `nearest_centroid_assignments.csv`: nearest-centroid result for each image
- `nearest_centroid_summary.csv`: per-label count and distance statistics
- `nearest_centroid_umap.png`: UMAP visualization with centroid markers
- `montage/*.jpg`: nearest samples for each manual label

For nested labels, increase `--label-depth`. For example, if labels are
`vendor_a/20pin`, use:

```bash
uv run connector-detection fit-centroids \
  configs/pipeline.example.toml \
  --labeled-image-dir data/labeled_pin_bands \
  --label-depth 2
```

Main outputs:

- `embeddings.npy`: final DINOv2 + weighted structural feature vectors
- `dinov2_embeddings.npy`: DINOv2-only embeddings
- `structural_features.npy`: raw structural features
- `structural_features_scaled.npy`: StandardScaler-transformed structural features
- `structural_features.csv`: readable structural summary per image
- `structural_feature_scaler.joblib`: fitted StandardScaler
- `manifest.csv`: image path index aligned with embeddings
- `embeddings_pca.npy`: PCA-reduced feature vectors
- `clusters.csv`: HDBSCAN cluster result and confidence
- `review/`: sampled images, montage images, and cluster feature statistics
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
