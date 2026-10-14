"""Embedding model wrapper.  Uses fastembed (ONNX, no PyTorch) so it fits in small
Render instances.  The model is loaded lazily and cached."""
import threading

from . import config


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBED_MODEL
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: list) -> list:
        if not texts:
            return []
        return [[float(x) for x in v] for v in self._load().embed(texts)]
