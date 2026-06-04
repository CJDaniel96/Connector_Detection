from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from connector_detection.images import list_images, resize_with_padding


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


def extract_embeddings(
    image_dir: Path,
    output_dir: Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    device: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(image_dir)
    if not paths:
        raise ValueError(f"No images found under {image_dir}")

    extractor = Dinov2FeatureExtractor(model_name, image_size, device)
    embeddings = extractor.encode_paths(paths, batch_size)

    embedding_path = output_dir / "embeddings.npy"
    manifest_path = output_dir / "manifest.csv"
    np.save(embedding_path, embeddings)
    pd.DataFrame({"image_path": [str(path) for path in paths]}).to_csv(
        manifest_path, index=False
    )
    return embedding_path, manifest_path
