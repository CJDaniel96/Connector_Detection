from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from connector_detection.images import list_images, resize_with_padding
from connector_detection.structural_features import (
    StructuralFeatureConfig,
    compute_structural_features,
    structural_feature_names,
)


class Dinov2FeatureExtractor:
    def __init__(self, model_name: str, image_size: int, device: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_paths(self, paths: list[Path], batch_size: int) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in tqdm(range(0, len(paths), batch_size), desc="DINOv2 embeddings"):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(resize_with_padding(image, self.image_size))

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
                cls = outputs.last_hidden_state[:, 0]
                cls = self.torch.nn.functional.normalize(cls, dim=1)
            embeddings.append(cls.cpu().numpy())

        return np.vstack(embeddings)


def build_combined_embeddings(
    paths: list[Path],
    output_dir: Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    structural_weight: float = 2.0,
    projection_profile_dims: int = 128,
    bright_threshold: float = 0.65,
    edge_threshold: float = 0.12,
    peak_threshold_std: float = 0.5,
    peak_min_distance: int = 3,
    structural_scaler: StandardScaler | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    if not paths:
        raise ValueError("No images provided for embedding extraction")

    structural_config = StructuralFeatureConfig(
        projection_dims=projection_profile_dims,
        bright_threshold=bright_threshold,
        edge_threshold=edge_threshold,
        peak_threshold_std=peak_threshold_std,
        peak_min_distance=peak_min_distance,
    )
    structural_features, structural_summary = compute_structural_features(
        paths,
        structural_config,
    )
    scaler = structural_scaler or StandardScaler()
    structural_scaled = (
        scaler.fit_transform(structural_features)
        if structural_scaler is None
        else scaler.transform(structural_features)
    )
    structural_weighted = structural_scaled * structural_weight

    extractor = Dinov2FeatureExtractor(model_name, image_size, device)
    dinov2_embeddings = extractor.encode_paths(paths, batch_size)
    embeddings = np.concatenate([dinov2_embeddings, structural_weighted], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "dinov2_embeddings.npy", dinov2_embeddings)
    np.save(output_dir / "structural_features.npy", structural_features)
    np.save(output_dir / "structural_features_scaled.npy", structural_scaled)
    np.save(output_dir / "embeddings.npy", embeddings)
    structural_summary.to_csv(output_dir / "structural_features.csv", index=False)
    (output_dir / "structural_feature_names.json").write_text(
        json.dumps(structural_feature_names(projection_profile_dims), indent=2)
    )
    return embeddings, structural_summary, scaler


def extract_embeddings(
    image_dir: Path,
    output_dir: Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    structural_weight: float = 2.0,
    projection_profile_dims: int = 128,
    bright_threshold: float = 0.65,
    edge_threshold: float = 0.12,
    peak_threshold_std: float = 0.5,
    peak_min_distance: int = 3,
    device: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(image_dir)
    if not paths:
        raise ValueError(f"No images found under {image_dir}")

    embeddings, structural_summary, scaler = build_combined_embeddings(
        paths=paths,
        output_dir=output_dir,
        model_name=model_name,
        image_size=image_size,
        batch_size=batch_size,
        structural_weight=structural_weight,
        projection_profile_dims=projection_profile_dims,
        bright_threshold=bright_threshold,
        edge_threshold=edge_threshold,
        peak_threshold_std=peak_threshold_std,
        peak_min_distance=peak_min_distance,
        device=device,
    )

    embedding_path = output_dir / "embeddings.npy"
    scaler_path = output_dir / "structural_feature_scaler.joblib"
    manifest_path = output_dir / "manifest.csv"
    joblib.dump(scaler, scaler_path)
    manifest = pd.DataFrame({"image_path": [str(path) for path in paths]})
    manifest = manifest.merge(structural_summary, on="image_path", how="left")
    manifest["structural_weight"] = structural_weight
    manifest.to_csv(manifest_path, index=False)
    return embedding_path, manifest_path
