"""Sentence-transformer embeddings. Heavy imports are deferred to first use,
so this module is importable in environments without torch installed."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from gcf_qna import config

if TYPE_CHECKING:
    import numpy as np


class Embedder:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32) -> "np.ndarray":
        import numpy as np
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return np.asarray(emb, dtype="float32")
