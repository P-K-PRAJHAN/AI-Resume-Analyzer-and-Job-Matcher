from typing import List
import os

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load SentenceTransformer. Ensure PyTorch-only dependencies are installed "
                "and conflicting TensorFlow/Keras packages are removed."
            ) from exc

    def encode_text(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(384, dtype="float32")
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.astype("float32")

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype="float32")
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.astype("float32")
