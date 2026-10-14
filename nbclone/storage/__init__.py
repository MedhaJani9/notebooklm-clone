from .base import Storage
from .local import LocalStorage


def get_storage(backend: str | None = None, data_dir: str | None = None) -> Storage:
    from .. import config

    backend = (backend or config.STORAGE_BACKEND).lower()
    if backend == "firestore":
        from .firestore_backend import FirestoreStorage
        return FirestoreStorage()
    return LocalStorage(data_dir or config.DATA_DIR)
