def test_notebook_crud(storage):
    nb = storage.create_notebook("A")
    assert storage.get_notebook(nb["id"])["name"] == "A"
    storage.rename_notebook(nb["id"], "B")
    assert [n["name"] for n in storage.list_notebooks()] == ["B"]
    storage.delete_notebook(nb["id"])
    assert storage.get_notebook(nb["id"]) is None


def test_data_is_scoped_per_notebook(storage):
    a, b = storage.create_notebook("A"), storage.create_notebook("B")
    storage.add_chunks(a["id"], [{"text": "x", "embedding": [0.1]}])
    storage.add_message(a["id"], {"role": "user", "content": "hi"})
    storage.save_artifact(a["id"], "report", "# R")
    assert storage.load_chunks(b["id"]) == []
    assert storage.load_messages(b["id"]) == []
    assert storage.list_artifacts(b["id"]) == []
    assert len(storage.load_chunks(a["id"])) == 1


def test_messages_and_artifacts_roundtrip(storage):
    nb = storage.create_notebook("A")["id"]
    storage.add_message(nb, {"role": "user", "content": "q"})
    storage.add_message(nb, {"role": "assistant", "content": "a"})
    assert [m["content"] for m in storage.load_messages(nb)] == ["q", "a"]
    art = storage.save_artifact(nb, "quiz", "# Quiz")
    assert storage.get_artifact(nb, art["id"])["markdown"] == "# Quiz"
    assert "markdown" not in storage.list_artifacts(nb)[0]


def test_delete_removes_everything(storage):
    nb = storage.create_notebook("A")["id"]
    storage.add_chunks(nb, [{"text": "x"}])
    storage.add_message(nb, {"role": "user", "content": "q"})
    storage.delete_notebook(nb)
    assert storage.list_notebooks() == []


def test_local_rejects_path_traversal(tmp_path):
    import pytest
    from nbclone.storage.local import LocalStorage
    with pytest.raises(ValueError):
        LocalStorage(str(tmp_path)).load_chunks("../../etc")


def test_local_persists_across_instances(tmp_path):
    from nbclone.storage.local import LocalStorage
    nb = LocalStorage(str(tmp_path)).create_notebook("Keep")["id"]
    assert LocalStorage(str(tmp_path)).get_notebook(nb)["name"] == "Keep"


def test_chunk_embeddings_roundtrip(storage):
    nb = storage.create_notebook("A")["id"]
    storage.add_chunks(nb, [{"text": "x", "page": 3, "embedding": [0.25, -0.5, 1.0]}])
    (c,) = storage.load_chunks(nb)
    assert c["text"] == "x" and c["page"] == 3 and c["embedding"] == [0.25, -0.5, 1.0]
