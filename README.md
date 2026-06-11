# Connector Detection

AI-based connector inspection research pipeline for connector crops, pin-band
crops, and image-level anomaly baseline comparison.

## Setup

```bash
uv sync
```

## Data Layout

PatchCore and DINO bank use the same folder contract so their validation results
can be compared fairly.

Training data uses one folder per class. Images directly under the class folder,
or under `good`, `ok`, `normal`, or `pass`, are treated as normal training
images.

```text
data/pin_band_train/
  20pin/
    good_001.png
    good_002.png
  24pin/
    good/
      good_101.png
```

Validation data uses the same class folders. Paths containing `good`, `ok`,
`normal`, or `pass` are OK. Paths containing `ng`, `bad`, `defect`, `dirty`,
`foreign`, `missing`, `shift`, `abnormal`, or `anomaly` are NG.

```text
data/pin_band_val/
  20pin/
    good/
      a.png
    defect/
      b.png
  24pin/
    good/
      c.png
    dirty/
      d.png
```

For nested labels such as `vendor_a/20pin`, pass `--class-depth 2`.

## Foundation Embedding Clustering

Use this workflow when images are not labeled yet and you want an initial
human-reviewable grouping.

```text
unlabeled images
-> DINOv2 / CLIP / ViT / ResNet embedding
-> PCA or UMAP reduction
-> K-Means clustering
-> UMAP visualization
-> representative images per cluster
-> manual cluster naming or merge notes
```

Example with DINOv2:

```bash
uv run connector-detection foundation-cluster \
  --image-dir data/unlabeled_pin_bands \
  --output-dir outputs/foundation_clusters \
  --model-kind dinov2 \
  --n-clusters 8 \
  --reducer pca \
  --reduced-dim 50 \
  --review-samples 30
```

Other supported backbones:

```bash
uv run connector-detection foundation-cluster --image-dir data/unlabeled --output-dir outputs/clip_clusters --model-kind clip --n-clusters 8
uv run connector-detection foundation-cluster --image-dir data/unlabeled --output-dir outputs/vit_clusters --model-kind vit --n-clusters 8
uv run connector-detection foundation-cluster --image-dir data/unlabeled --output-dir outputs/resnet_clusters --model-kind resnet --n-clusters 8
```

Main outputs:

- `embeddings.npy`: raw foundation model image embeddings.
- `embeddings_reduced.npy`: PCA or UMAP features used by K-Means.
- `clusters.csv`: per-image cluster id, centroid distance, and representative rank.
- `umap_clusters.png`: 2D UMAP plot colored by K-Means cluster.
- `review/cluster_*/`: representative images nearest to each cluster centroid.
- `review/montage/cluster_*.jpg`: montage for fast manual review.
- `cluster_labels_template.csv`: fill `cluster_name` and optional `merge_to` after review.
- `foundation_clustering_report.md`: compact run summary.

After reviewing the montage folders, edit `cluster_labels_template.csv`. To name
cluster 0 as `20pin` and merge cluster 3 into cluster 0, set `cluster_name=20pin`
for cluster 0 and `merge_to=0` for cluster 3. Then apply the review decisions:

```bash
uv run connector-detection apply-foundation-cluster-labels \
  --clusters-csv outputs/foundation_clusters/clusters.csv \
  --labels-csv outputs/foundation_clusters/cluster_labels_template.csv \
  --output outputs/foundation_clusters/clusters_named.csv
```

Splitting a cluster is intentionally left as a manual per-image review step:
write notes in `split_note`, then move those images or create a corrected CSV
based on `clusters.csv`.

## PatchCore Baseline

PatchCore is a standalone anomalib baseline. It uses anomalib `Folder`,
`Patchcore`, and `Engine`; DINOv2 and structural features are not used.

```bash
uv run connector-detection patchcore-train \
  configs/patchcore.example.toml \
  --train-image-dir data/pin_band_train \
  --validation-image-dir data/pin_band_val \
  --output-dir outputs/patchcore_pin_bands
```

Validate an existing model:

```bash
uv run connector-detection patchcore-validate \
  configs/patchcore.example.toml \
  outputs/patchcore_pin_bands/patchcore_models.joblib \
  --validation-image-dir data/pin_band_val \
  --output-dir outputs/patchcore_validation
```

Validate only one class:

```bash
uv run connector-detection patchcore-validate \
  configs/patchcore.example.toml \
  outputs/patchcore_pin_bands/patchcore_models.joblib \
  --validation-image-dir data/pin_band_val \
  --output-dir outputs/patchcore_validation_20pin \
  --class-label 20pin
```

Main outputs:

- `patchcore_models.joblib`: per-class anomalib checkpoint index.
- `patchcore_anomalib_summary.csv`: per-class anomalib metrics and artifact paths.
- `patchcore_report.md`: thin project-level summary.
- `classes/*/anomalib`: anomalib trainer logs, checkpoints, and visual artifacts.
- `classes/*/predictions.csv`: prediction scores when anomalib returns prediction batches.

Legacy aliases still work: `train-patchcore` and `validate-patchcore`.

## DINOv2 + Structural Bank Baseline

DINO bank is a separate image-level anomaly baseline. It builds one normal
feature bank per class:

```text
pin-band crop
-> DINOv2 CLS embedding
-> structural features
-> concat
-> StandardScaler
-> PCA
-> per-class nearest-neighbor bank
-> anomaly score
```

Train and optionally validate:

```bash
uv run connector-detection dinobank-train \
  configs/dinobank.example.toml \
  --train-image-dir data/pin_band_train \
  --validation-image-dir data/pin_band_val \
  --output-dir outputs/dinobank_pin_bands
```

Validate an existing model:

```bash
uv run connector-detection dinobank-validate \
  configs/dinobank.example.toml \
  outputs/dinobank_pin_bands/dinobank_model.joblib \
  --validation-image-dir data/pin_band_val \
  --output-dir outputs/dinobank_validation
```

Main outputs:

- `dinobank_model.joblib`: per-class feature banks, scalers, PCA, thresholds.
- `dinobank_summary.csv`: per-class train distance statistics and thresholds.
- `validation/predictions.csv`: image-level anomaly score and OK/NG prediction.
- `validation/dinobank_validation_summary.csv`: per-class validation metrics.
- `validation/plots/score_histogram.png`: OK/NG score distribution.
- `validation/plots/validation_umap.png`: UMAP review plot when enough samples exist.
- `validation/montage/top_anomaly_scores.jpg`: highest-score validation samples.

Thresholds are set from normal training images using leave-one-out nearest
distance quantiles. The default is `dinobank_threshold_quantile = 0.995`.

## Dinomaly Baseline

Dinomaly is a reconstruction-style Transformer anomaly model. In this project
it is trained as one unified multi-class anomalib model over all selected
classes, matching Dinomaly's main multi-class UAD setting.

Train and optionally validate:

```bash
uv run connector-detection dinomaly-train \
  configs/dinomaly.example.toml \
  --train-image-dir data/pin_band_train \
  --validation-image-dir data/pin_band_val \
  --output-dir outputs/dinomaly_pin_bands
```

Validate an existing model:

```bash
uv run connector-detection dinomaly-validate \
  configs/dinomaly.example.toml \
  outputs/dinomaly_pin_bands/dinomaly_model.joblib \
  --validation-image-dir data/pin_band_val \
  --output-dir outputs/dinomaly_validation
```

Run blind prediction on unlabeled images:

```bash
uv run connector-detection dinomaly-predict \
  configs/dinomaly.example.toml \
  outputs/dinomaly_pin_bands/dinomaly_model.joblib \
  --image-dir data/blind_pin_bands \
  --output-dir outputs/dinomaly_blind
```

Blind prediction outputs:

- `blind_predictions.csv`: per-image score, OK/NG prediction, and heatmap paths.
- `classified/OK`: original images predicted OK.
- `classified/NG`: original images predicted NG.
- `classified/OK_overlays` / `classified/NG_overlays`: copied overlay heatmaps.
- `analysis/prediction_counts.png`: OK/NG count chart.
- `analysis/score_histogram_by_prediction.png`: anomaly score distribution.
- `analysis/top_anomaly_scores.png`: highest anomaly score bar chart.

Use only selected classes:

```bash
uv run connector-detection dinomaly-train \
  configs/dinomaly.example.toml \
  --train-image-dir data/pin_band_train \
  --validation-image-dir data/pin_band_val \
  --class-label 20pin \
  --class-label 24pin
```

Main outputs:

- `dinomaly_model.joblib`: checkpoint index, config, and class labels.
- `dinomaly_anomalib_summary.csv`: anomalib metrics and prediction artifact paths.
- `dinomaly_report.md`: compact training or validation report.
- `anomalib/`: anomalib trainer logs, checkpoints, and visual artifacts.
- `predictions.csv`: prediction scores when anomalib returns prediction batches.
- `heatmaps/*_raw_heatmap.png`: raw anomaly heatmaps.
- `heatmaps/*_overlay.png`: anomaly heatmaps overlaid on the original image.
- `plots/score_histogram.png`: anomaly score distribution.
- `montage/top_anomaly_scores.jpg`: highest-score validation images.
- `montage/top_anomaly_overlays.jpg`: highest-score validation overlays.

For long pin-band crops, `configs/dinomaly.example.toml` uses
`dinomaly_image_size = [112, 560]` and `dinomaly_crop_size = "null"`. In this
project, `"null"` means resize + ImageNet normalization only, without anomalib's
default square `CenterCrop(392)`.

## Compare Baselines

After running both baselines on the same validation root:

```bash
uv run connector-detection compare-baselines \
  --patchcore-predictions outputs/patchcore_validation \
  --dinobank-predictions outputs/dinobank_validation \
  --output-dir outputs/baseline_comparison
```

Outputs:

- `baseline_comparison.csv`: per-image PatchCore and DINO bank scores aligned by path.
- `baseline_metrics.csv`: accuracy, precision, recall, F1, and ROC AUC when labels exist.
- `baseline_score_histogram.png`: score distributions for both baselines.
- `baseline_comparison_report.md`: compact comparison report.

## Clustering And Review Utilities

The original DINOv2 + structural clustering tools remain available for
connector or pin-band grouping:

```bash
uv run connector-detection extract configs/pipeline.example.toml
uv run connector-detection cluster configs/pipeline.example.toml
uv run connector-detection review configs/pipeline.example.toml
uv run connector-detection umap configs/pipeline.example.toml
```

Run all clustering stages:

```bash
uv run connector-detection run configs/pipeline.example.toml
```

## Crop Pin Bands From PASCAL VOC XML

If you already have connector images and pin-band PASCAL VOC XML labels, crop
pin bands and rotate every crop into the same `up` orientation:

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

## Rotate Full Images From PASCAL VOC Box Position

To rotate the whole source image according to the center position of a specific
VOC class, use `rotate-images-by-voc`. For example, this rotates images so the
selected `pin_band` bbox orientation becomes `up`:

```bash
uv run connector-detection rotate-images-by-voc \
  --xml-dir data/pin_band_xml \
  --image-dir data/connector_images \
  --output-dir outputs/rotated_connector_images \
  --label pin_band \
  --target-orientation up
```

If one image has multiple matching boxes, one rotated image is written per box.
The output manifest is `outputs/rotated_connector_images/rotated_images_manifest.csv`.

## Nearest-Centroid Group Assignment

For manual class assignment rather than anomaly detection:

```bash
uv run connector-detection fit-centroids \
  configs/pipeline.example.toml \
  --labeled-image-dir data/labeled_pin_bands \
  --output-dir outputs/nearest_centroid_pin_groups
```

Then assign new embeddings:

```bash
uv run connector-detection assign-centroids \
  outputs/nearest_centroid_pin_groups/embeddings.npy \
  outputs/nearest_centroid_pin_groups/manifest.csv \
  outputs/nearest_centroid_pin_groups/nearest_centroid_model.joblib \
  outputs/nearest_centroid_predictions.csv
```
