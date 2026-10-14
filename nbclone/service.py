"""NotebookService: orchestrates storage, ingestion, retrieval, chat and artifacts.

The UI layer (app.py) only calls this class, keeping the UI free of business logic.
"""
import threading

from . import artifacts, config, ingestion, rag
from .base_types import ValidationError
from .retrieval import NotebookIndex
from .storage.base import new_id, now


class NotebookService:
    def __init__(self, storage, embedder, llm):
        self.storage, self.embedder, self.llm = storage, embedder, llm
        self._indexes = {}
        self._lock = threading.Lock()

    # ---------- notebooks ----------
    @staticmethod
    def _clean_name(name: str) -> str:
        name = (name or "").strip()
        if not name:
            raise ValidationError("Please enter a notebook name.")
        return name[:80]

    def create_notebook(self, name):
        return self.storage.create_notebook(self._clean_name(name))

    def rename_notebook(self, nb_id, name):
        self.storage.rename_notebook(nb_id, self._clean_name(name))

    def delete_notebook(self, nb_id):
        self.storage.delete_notebook(nb_id)
        self._indexes.pop(nb_id, None)

    def list_notebooks(self):
        return self.storage.list_notebooks()

    def _require(self, nb_id):
        if not nb_id or not self.storage.get_notebook(nb_id):
            raise ValidationError("Select or create a notebook first.")

    # ---------- sources ----------
    def _reject_duplicate(self, nb_id, name):
        if any(s["name"] == name for s in self.storage.list_sources(nb_id)):
            raise ValidationError(f"'{name[:80]}' is already in this notebook.")

    def _ingest(self, nb_id, name, stype, segments, raw=None):
        self._reject_duplicate(nb_id, name)
        sid = new_id()
        chunks = ingestion.chunk_segments(segments, sid, name)
        embeddings = self.embedder.embed([c["text"] for c in chunks])
        for c, e in zip(chunks, embeddings):
            c["embedding"] = e
        if raw is not None:
            self.storage.save_raw(nb_id, name, raw)
        self.storage.save_extracted(nb_id, sid, "\n\n".join(s["text"] for s in segments))
        self.storage.add_chunks(nb_id, chunks)
        src = {"id": sid, "name": name, "type": stype, "chunks": len(chunks), "created_at": now()}
        self.storage.add_source(nb_id, src)
        self._indexes.pop(nb_id, None)  # force rebuild with the new chunks
        return src

    def add_file(self, nb_id, filename, data):
        self._require(nb_id)
        self._reject_duplicate(nb_id, filename)  # before the (slow) extraction
        segments = ingestion.extract_file(filename, data)
        stype = filename.rsplit(".", 1)[-1].lower()
        return self._ingest(nb_id, filename, stype, segments, raw=data)

    def add_url(self, nb_id, url):
        self._require(nb_id)
        title, segments = ingestion.extract_url(url)
        return self._ingest(nb_id, f"{title} ({url.strip()})"[:200], "url", segments)

    def list_sources(self, nb_id):
        return self.storage.list_sources(nb_id) if nb_id else []

    # ---------- retrieval / chat ----------
    def index(self, nb_id) -> NotebookIndex:
        with self._lock:
            if nb_id not in self._indexes:
                self._indexes[nb_id] = NotebookIndex(nb_id, self.storage.load_chunks(nb_id), self.embedder)
            return self._indexes[nb_id]

    def search(self, nb_id, query, k=None, mode="hybrid"):
        self._require(nb_id)
        return self.index(nb_id).search(query, k or config.TOP_K, mode)

    def chat(self, nb_id, question, mode="hybrid"):
        self._require(nb_id)
        question = (question or "").strip()
        if not question:
            raise ValidationError("Please type a question.")
        history = self.storage.load_messages(nb_id)
        result = rag.answer(question, self.index(nb_id), self.llm, mode=mode, k=config.TOP_K, history=history)
        content = result["answer"] + (rag.format_sources_footer(result["citations"]) if result["citations"] else "")
        self.storage.add_message(nb_id, {"role": "user", "content": question})
        self.storage.add_message(nb_id, {"role": "assistant", "content": content, "citations": result["citations"],
                                         "mode": mode})
        return {**result, "content": content}

    def history(self, nb_id):
        return self.storage.load_messages(nb_id) if nb_id else []

    # ---------- artifacts ----------
    def generate_artifact(self, nb_id, kind):
        self._require(nb_id)
        md = artifacts.generate(kind, self.storage.load_chunks(nb_id), self.llm)
        meta = self.storage.save_artifact(nb_id, kind, md)
        return {**meta, "markdown": md}

    def list_artifacts(self, nb_id):
        return self.storage.list_artifacts(nb_id) if nb_id else []

    def get_artifact(self, nb_id, artifact_id):
        return self.storage.get_artifact(nb_id, artifact_id)
