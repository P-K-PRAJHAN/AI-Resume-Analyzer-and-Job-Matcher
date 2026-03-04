from typing import Dict, List

import faiss
import numpy as np


class ResumeVectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.resume_records: List[Dict] = []

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        if embeddings.size == 0:
            return
        self.index.reset()
        self.index.add(embeddings.astype("float32"))
        self.resume_records = metadata

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = dict(self.resume_records[idx])
            record["vector_similarity"] = float(score)
            results.append(record)
        return results
