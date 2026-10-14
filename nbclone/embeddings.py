"""Embedding model wrapper.  Uses fastembed (ONNX, no PyTorch) so it fits in small
Render instances.  The model is loaded lazily and cached.

Memory: with fastembed's defaults (batch 256, one ONNX thread per host core, memory
arena on) embedding a ~500-chunk PDF peaked at 2.3 GB and got the 512 MB Render
instance OOM-killed.  Small batches, 1 thread and no arena peak at ~360 MB (measured)."""
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
                self._model = TextEmbedding(model_name=self.model_name, threads=config.EMBED_THREADS,
                                            enable_cpu_mem_arena=False)
        return self._model

    def embed(self, texts: list) -> list:
        if not texts:
            return []
        return [[float(x) for x in v] for v in self._load().embed(texts, batch_size=config.EMBED_BATCH)]
