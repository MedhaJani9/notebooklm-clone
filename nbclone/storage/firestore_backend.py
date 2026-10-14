"""Firestore backend (data survives Render restarts/redeploys).

Data model (one document per record, each < 1 MiB):
    notebooks/{id}                       name, created_at
    notebooks/{id}/sources/{sid}         filename|url, type, chunks
    notebooks/{id}/chunks/{seq}          text, source, page, embedding, ...
    notebooks/{id}/messages/{seq}        role, content, citations, ts
    notebooks/{id}/artifacts/{aid}       type, markdown, created_at

Embeddings are stored as packed float32 bytes, not arrays: Firestore indexes every
array element, which made writing a 124-chunk PDF take ~13 s instead of ~1 s.

Raw uploads are NOT persisted here (Firestore is not a blob store); the
extracted text lives in the chunk documents, which is all RAG needs.
"""
import json
import os
from array import array

from .base import Storage, new_id, now

SUBS = ("sources", "chunks", "messages", "artifacts")


def _pack(chunk):
    return {**chunk, "embedding": array("f", chunk["embedding"]).tobytes()} if "embedding" in chunk else chunk


def _unpack(chunk):
    e = chunk.get("embedding")
    if isinstance(e, bytes):  # older docs may still hold plain lists
        chunk["embedding"] = array("f", e).tolist()
    return chunk


def make_client():
    """Build a client from GOOGLE_CREDENTIALS_JSON (service-account JSON string)."""
    from google.cloud import firestore
    from google.oauth2 import service_account

    raw = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if raw:
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(info)
        return firestore.Client(project=info.get("project_id"), credentials=creds)
    return firestore.Client()  # Application Default Credentials (e.g. Cloud Run)


class FirestoreStorage(Storage):
    def __init__(self, client=None):
        self.db = client or make_client()

    def _nb(self, nb_id):
        if not nb_id:
            raise ValueError("Invalid notebook id.")
        return self.db.collection("notebooks").document(nb_id)

    def _append(self, nb_id, sub, data, doc_id=None):
        self._nb(nb_id).collection(sub).document(doc_id or new_id()).set(data)

    def _all(self, nb_id, sub):
        return [d.to_dict() for d in self._nb(nb_id).collection(sub).stream()]

    # notebooks
    def create_notebook(self, name):
        nb = {"id": new_id(), "name": name, "created_at": now()}
        self._nb(nb["id"]).set(nb)
        return nb

    def list_notebooks(self):
        items = [d.to_dict() for d in self.db.collection("notebooks").stream()]
        return sorted(items, key=lambda n: n["created_at"])

    def get_notebook(self, nb_id):
        snap = self._nb(nb_id).get()
        return snap.to_dict() if snap.exists else None

    def rename_notebook(self, nb_id, name):
        if not self.get_notebook(nb_id):
            raise ValueError("Notebook not found.")
        self._nb(nb_id).update({"name": name})

    def delete_notebook(self, nb_id):
        for sub in SUBS:
            batch, n = self.db.batch(), 0
            for d in self._nb(nb_id).collection(sub).stream():
                batch.delete(d.reference)
                n += 1
                if n % 400 == 0:
                    batch.commit()
                    batch = self.db.batch()
            batch.commit()
        self._nb(nb_id).delete()

    # sources
    def add_source(self, nb_id, source):
        self._append(nb_id, "sources", source, source["id"])

    def list_sources(self, nb_id):
        return sorted(self._all(nb_id, "sources"), key=lambda s: s.get("created_at", 0))

    def save_raw(self, nb_id, filename, data):
        pass  # not persisted in Firestore (see module docstring)

    def save_extracted(self, nb_id, source_id, text):
        pass  # text is persisted as chunks

    # chunks
    def add_chunks(self, nb_id, chunks):
        col = self._nb(nb_id).collection("chunks")
        start = len(list(col.stream()))
        batch = self.db.batch()
        for i, c in enumerate(chunks):
            batch.set(col.document(f"{start + i:08d}"), {**_pack(c), "seq": start + i})
            if (i + 1) % 400 == 0:  # Firestore batches are limited to 500 writes
                batch.commit()
                batch = self.db.batch()
        batch.commit()

    def load_chunks(self, nb_id):
        return sorted((_unpack(c) for c in self._all(nb_id, "chunks")), key=lambda c: c["seq"])

    # messages
    def add_message(self, nb_id, message):
        ts = now()
        self._append(nb_id, "messages", {**message, "ts": ts}, f"{int(ts * 1000):015d}_{new_id()[:4]}")

    def load_messages(self, nb_id):
        return sorted(self._all(nb_id, "messages"), key=lambda m: m["ts"])

    # artifacts
    def save_artifact(self, nb_id, kind, markdown):
        art = {"id": new_id(), "type": kind, "created_at": now()}
        self._append(nb_id, "artifacts", {**art, "markdown": markdown}, art["id"])
        return art

    def list_artifacts(self, nb_id):
        items = [{k: v for k, v in a.items() if k != "markdown"} for a in self._all(nb_id, "artifacts")]
        return sorted(items, key=lambda a: a["created_at"])

    def get_artifact(self, nb_id, artifact_id):
        snap = self._nb(nb_id).collection("artifacts").document(artifact_id).get()
        return snap.to_dict() if snap.exists else None
