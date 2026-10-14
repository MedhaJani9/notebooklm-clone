import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nbclone.retrieval import tokenize  # noqa: E402
from nbclone.service import NotebookService  # noqa: E402
from nbclone.storage.firestore_backend import FirestoreStorage  # noqa: E402
from nbclone.storage.local import LocalStorage  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download needed)."""
    DIM = 256

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in tokenize(t):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, temperature=0.2, retries=3):
        self.prompts.append(prompt)
        if "Answer Key" in prompt or "quiz in Markdown" in prompt:
            return "# Quiz\n1. Q?\n\n## Answer Key\n1. A"
        if "study report" in prompt:
            return "# Report\n## Executive Summary\nText\n## Sources Referenced\n- x"
        return "Lists are mutable containers [1]."


# ---- minimal in-memory Firestore fake ----
class _Snap:
    def __init__(self, ref, data):
        self.reference, self.id, self._d = ref, ref.id, data
        self.exists = data is not None

    def to_dict(self):
        return dict(self._d) if self._d is not None else None


class _Doc:
    def __init__(self, store, path):
        self.store, self.path, self.id = store, path, path[-1]

    def set(self, data):
        self.store.docs[self.path] = dict(data)

    def update(self, data):
        self.store.docs[self.path].update(data)

    def get(self):
        return _Snap(self, self.store.docs.get(self.path))

    def delete(self):
        self.store.docs.pop(self.path, None)

    def collection(self, name):
        return _Col(self.store, self.path + (name,))


class _Col:
    def __init__(self, store, path):
        self.store, self.path = store, path

    def document(self, doc_id):
        return _Doc(self.store, self.path + (doc_id,))

    def stream(self):
        n = len(self.path)
        return [_Snap(_Doc(self.store, p), d) for p, d in list(self.store.docs.items())
                if len(p) == n + 1 and p[:n] == self.path]


class _Batch:
    def __init__(self):
        self.ops = []

    def set(self, ref, data):
        self.ops.append(lambda: ref.set(data))

    def delete(self, ref):
        self.ops.append(ref.delete)

    def commit(self):
        for op in self.ops:
            op()
        self.ops = []


class FakeFirestore:
    def __init__(self):
        self.docs = {}

    def collection(self, name):
        return _Col(self, (name,))

    def batch(self):
        return _Batch()


@pytest.fixture(params=["local", "firestore"])
def storage(request, tmp_path):
    return LocalStorage(str(tmp_path)) if request.param == "local" else FirestoreStorage(FakeFirestore())


@pytest.fixture
def service(storage):
    return NotebookService(storage, FakeEmbedder(), FakeLLM())
