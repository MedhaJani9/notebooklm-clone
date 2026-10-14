"""Local-disk backend.  Layout (one folder per notebook id):

data/notebooks/<id>/
    meta.json            notebook name, created_at
    sources.json         list of sources (filename/url, type, chunk count)
    raw/                 original uploaded files
    extracted/<src>.txt  extracted plain text per source
    chunks.json          chunks + embeddings (vector index is rebuilt from this)
    messages.json        chat history
    artifacts/<id>.md    generated reports / quizzes
"""
import json
import os
import re
import shutil
import threading

from .base import Storage, new_id, now


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name))[:120] or "file"


class LocalStorage(Storage):
    def __init__(self, root: str):
        self.root = os.path.join(root, "notebooks")
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.Lock()

    # helpers
    def _dir(self, nb_id: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{12}", nb_id or ""):  # blocks path traversal
            raise ValueError("Invalid notebook id.")
        return os.path.join(self.root, nb_id)

    def _read(self, nb_id, fname, default):
        path = os.path.join(self._dir(nb_id), fname)
        if not os.path.exists(path):
            return default
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, nb_id, fname, obj):
        path = os.path.join(self._dir(nb_id), fname)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)

    # notebooks
    def create_notebook(self, name):
        nb = {"id": new_id(), "name": name, "created_at": now()}
        d = self._dir(nb["id"])
        for sub in ("raw", "extracted", "artifacts"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        self._write(nb["id"], "meta.json", nb)
        return nb

    def list_notebooks(self):
        out = []
        for nb_id in os.listdir(self.root):
            if re.fullmatch(r"[a-f0-9]{12}", nb_id):
                meta = self._read(nb_id, "meta.json", None)
                if meta:
                    out.append(meta)
        return sorted(out, key=lambda n: n["created_at"])

    def get_notebook(self, nb_id):
        return self._read(nb_id, "meta.json", None) if os.path.isdir(self._dir(nb_id)) else None

    def rename_notebook(self, nb_id, name):
        meta = self.get_notebook(nb_id)
        if not meta:
            raise ValueError("Notebook not found.")
        meta["name"] = name
        self._write(nb_id, "meta.json", meta)

    def delete_notebook(self, nb_id):
        shutil.rmtree(self._dir(nb_id), ignore_errors=True)

    # sources
    def add_source(self, nb_id, source):
        with self._lock:
            items = self._read(nb_id, "sources.json", [])
            items.append(source)
            self._write(nb_id, "sources.json", items)

    def list_sources(self, nb_id):
        return self._read(nb_id, "sources.json", [])

    def save_raw(self, nb_id, filename, data):
        with open(os.path.join(self._dir(nb_id), "raw", _safe_name(filename)), "wb") as f:
            f.write(data)

    def save_extracted(self, nb_id, source_id, text):
        with open(os.path.join(self._dir(nb_id), "extracted", _safe_name(source_id) + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)

    # chunks
    def add_chunks(self, nb_id, chunks):
        with self._lock:
            items = self._read(nb_id, "chunks.json", [])
            items.extend(chunks)
            self._write(nb_id, "chunks.json", items)

    def load_chunks(self, nb_id):
        return self._read(nb_id, "chunks.json", [])

    # messages
    def add_message(self, nb_id, message):
        with self._lock:
            items = self._read(nb_id, "messages.json", [])
            items.append({**message, "ts": now()})
            self._write(nb_id, "messages.json", items)

    def load_messages(self, nb_id):
        return self._read(nb_id, "messages.json", [])

    # artifacts
    def save_artifact(self, nb_id, kind, markdown):
        art = {"id": new_id(), "type": kind, "created_at": now()}
        with open(os.path.join(self._dir(nb_id), "artifacts", f"{art['id']}.md"), "w", encoding="utf-8") as f:
            f.write(markdown)
        with self._lock:
            items = self._read(nb_id, "artifacts.json", [])
            items.append(art)
            self._write(nb_id, "artifacts.json", items)
        return art

    def list_artifacts(self, nb_id):
        return self._read(nb_id, "artifacts.json", [])

    def get_artifact(self, nb_id, artifact_id):
        meta = next((a for a in self.list_artifacts(nb_id) if a["id"] == artifact_id), None)
        if not meta:
            return None
        with open(os.path.join(self._dir(nb_id), "artifacts", f"{artifact_id}.md"), encoding="utf-8") as f:
            return {**meta, "markdown": f.read()}
