"""Storage abstraction layer.

The rest of the app only talks to this interface, so the persistence backend
(local disk or Firestore) can be swapped with the STORAGE_BACKEND env var.
Every record is keyed by a notebook id so sources, chats and artifacts always
belong to exactly one notebook.
"""
from abc import ABC, abstractmethod
import time
import uuid


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> float:
    return time.time()


class Storage(ABC):
    # ---- notebooks ----
    @abstractmethod
    def create_notebook(self, name: str) -> dict: ...
    @abstractmethod
    def list_notebooks(self) -> list: ...
    @abstractmethod
    def get_notebook(self, nb_id: str) -> dict | None: ...
    @abstractmethod
    def rename_notebook(self, nb_id: str, name: str) -> None: ...
    @abstractmethod
    def delete_notebook(self, nb_id: str) -> None: ...

    # ---- sources / raw files / extracted text ----
    @abstractmethod
    def add_source(self, nb_id: str, source: dict) -> None: ...
    @abstractmethod
    def list_sources(self, nb_id: str) -> list: ...
    @abstractmethod
    def save_raw(self, nb_id: str, filename: str, data: bytes) -> None: ...
    @abstractmethod
    def save_extracted(self, nb_id: str, source_id: str, text: str) -> None: ...

    # ---- chunks (text + embedding + citation metadata) ----
    @abstractmethod
    def add_chunks(self, nb_id: str, chunks: list) -> None: ...
    @abstractmethod
    def load_chunks(self, nb_id: str) -> list: ...

    # ---- chat history ----
    @abstractmethod
    def add_message(self, nb_id: str, message: dict) -> None: ...
    @abstractmethod
    def load_messages(self, nb_id: str) -> list: ...

    # ---- artifacts (.md) ----
    @abstractmethod
    def save_artifact(self, nb_id: str, kind: str, markdown: str) -> dict: ...
    @abstractmethod
    def list_artifacts(self, nb_id: str) -> list: ...
    @abstractmethod
    def get_artifact(self, nb_id: str, artifact_id: str) -> dict | None: ...
