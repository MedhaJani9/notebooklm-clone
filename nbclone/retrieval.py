"""Per-notebook retrieval index with two methods.

* "vector": cosine similarity over ChromaDB (dense embeddings).
* "hybrid": dense search + BM25 keyword search merged with Reciprocal Rank
  Fusion (RRF).  Catches exact terms/names that embeddings can blur.

The index lives in memory and is rebuilt from persisted chunks on demand, so it
survives restarts regardless of the storage backend.
"""
import re

import chromadb
from rank_bm25 import BM25Okapi

RRF_K = 60
_client = chromadb.EphemeralClient()


def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


class NotebookIndex:
    def __init__(self, nb_id: str, chunks: list, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.collection = None
        self.bm25 = None
        if not chunks:
            return
        name = f"nb_{nb_id}"
        try:
            _client.delete_collection(name)
        except Exception:
            pass
        self.collection = _client.create_collection(name, metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), 500):  # Chroma add() batches
            part = chunks[i:i + 500]
            self.collection.add(
                ids=[str(i + j) for j in range(len(part))],
                embeddings=[c["embedding"] for c in part],
                documents=[c["text"] for c in part],
                metadatas=[{"source": c["source"], "page": c["page"] or 0} for c in part])
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def _vector(self, query: str, n: int) -> list:
        q = self.embedder.embed([query])[0]
        res = self.collection.query(query_embeddings=[q], n_results=min(n, len(self.chunks)))
        return [(int(i), 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def _bm25(self, query: str, n: int) -> list:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]

    def search(self, query: str, k: int = 5, mode: str = "vector") -> list:
        if not self.chunks:
            return []
        if mode == "hybrid":
            fused = {}
            for ranked in (self._vector(query, 20), self._bm25(query, 20)):
                for rank, (i, _) in enumerate(ranked):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        elif mode == "vector":
            top = self._vector(query, k)
        else:
            raise ValueError(f"Unknown retrieval mode '{mode}'.")
        return [{**{key: v for key, v in self.chunks[i].items() if key != "embedding"},
                 "score": round(score, 4), "method": mode} for i, score in top]
