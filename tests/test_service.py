import pytest

from nbclone.base_types import ValidationError
from nbclone.service import NotebookService
from tests.conftest import FakeEmbedder, FakeLLM

TEXT = (b"Python lists are mutable ordered containers. Use append to add items. " * 30
        + b"\n\nTuples are immutable sequences that cannot change after creation. " * 30)


def test_full_flow_and_persistence(service, storage):
    nb = service.create_notebook("Study")["id"]
    src = service.add_file(nb, "notes.txt", TEXT)
    assert src["chunks"] > 1 and service.list_sources(nb)[0]["name"] == "notes.txt"

    res = service.chat(nb, "What are python lists?", mode="vector")
    assert res["citations"] and res["citations"][0]["cited"] and "**Sources:**" in res["content"]

    # "restart": new service on same storage still has history and can answer
    svc2 = NotebookService(storage, FakeEmbedder(), FakeLLM())
    assert [m["role"] for m in svc2.history(nb)] == ["user", "assistant"]
    assert svc2.search(nb, "tuples immutable", k=1, mode="hybrid")[0]["source"] == "notes.txt"

    report = svc2.generate_artifact(nb, "report")
    quiz = svc2.generate_artifact(nb, "quiz")
    assert report["markdown"].startswith("# Report") and "Answer Key" in quiz["markdown"]
    assert {a["type"] for a in svc2.list_artifacts(nb)} == {"report", "quiz"}


def test_notebooks_are_isolated(service):
    a, b = service.create_notebook("A")["id"], service.create_notebook("B")["id"]
    service.add_file(a, "a.txt", TEXT)
    assert service.search(b, "python lists") == []
    assert service.history(b) == []


def test_validation_errors(service):
    with pytest.raises(ValidationError):
        service.create_notebook("   ")
    with pytest.raises(ValidationError):
        service.chat(None, "hi")
    nb = service.create_notebook("A")["id"]
    with pytest.raises(ValidationError):
        service.chat(nb, "  ")
    with pytest.raises(ValueError):
        service.generate_artifact(nb, "report")  # no sources yet


def test_chat_without_sources_is_friendly(service):
    nb = service.create_notebook("A")["id"]
    assert "no sources" in service.chat(nb, "hi")["answer"].lower()


def test_rename_delete_switch(service):
    a, b = service.create_notebook("A")["id"], service.create_notebook("B")["id"]
    service.rename_notebook(a, "Renamed")
    service.delete_notebook(b)
    assert [n["name"] for n in service.list_notebooks()] == ["Renamed"]


def test_duplicate_source_rejected(service):
    nb = service.create_notebook("A")["id"]
    service.add_file(nb, "notes.txt", TEXT)
    with pytest.raises(ValidationError, match="already in this notebook"):
        service.add_file(nb, "notes.txt", TEXT)
    assert len(service.list_sources(nb)) == 1
