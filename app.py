"""Gradio front end for the NotebookLM-style app.  All logic lives in nbclone.service."""
import functools
import logging
import os
import tempfile
import threading

import gradio as gr

from nbclone.embeddings import Embedder
from nbclone.ingestion import IngestionError
from nbclone.llm import LLMClient, LLMError
from nbclone.service import NotebookService
from nbclone.storage import get_storage

log = logging.getLogger("app")


def friendly(fn):
    """Turn exceptions into readable error toasts instead of stack traces."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except (ValueError, IngestionError, LLMError) as e:  # ValidationError is a ValueError
            raise gr.Error(str(e))
        except Exception:
            log.exception("Unexpected error in %s", fn.__name__)
            raise gr.Error("Something went wrong on the server. Please try again.")
    return wrapper


def build_app(svc: NotebookService) -> gr.Blocks:
    def nb_update(selected=None):
        nbs = svc.list_notebooks()
        return gr.update(choices=[(n["name"], n["id"]) for n in nbs],
                         value=selected if selected else (nbs[0]["id"] if nbs else None))

    def sources_md(nb_id):
        src = svc.list_sources(nb_id)
        if not src:
            return "_No sources yet. Upload a PDF/PPTX/TXT or add a URL._"
        return "\n".join(f"- **{s['name']}** ({s['type']}, {s['chunks']} chunks)" for s in src)

    def chat_history(nb_id):
        return [{"role": m["role"], "content": m["content"]} for m in svc.history(nb_id)]

    def art_update(nb_id, selected=None):
        arts = svc.list_artifacts(nb_id)
        choices = [(f"{a['type']} · {a['id']}", a["id"]) for a in reversed(arts)]
        return gr.update(choices=choices, value=selected)

    def panel(nb_id):
        """Everything that depends on the selected notebook."""
        return sources_md(nb_id), chat_history(nb_id), "", art_update(nb_id), "", None

    def download_file(artifact):
        path = os.path.join(tempfile.gettempdir(), f"{artifact['type']}_{artifact['id']}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(artifact["markdown"])
        return path

    def citations_md(res):
        if not res["citations"]:
            return ""
        lines = [f"**Retrieval** ({res['chunks'][0]['method']}): {len(res['chunks'])} chunks in "
                 f"{res['retrieval_time']:.2f}s · LLM {res['llm_time']:.1f}s\n"]
        for c in res["citations"]:
            loc = f", {c['label']}" if c["label"] else ""
            mark = "✅ cited" if c["cited"] else "retrieved"
            lines.append(f"**[{c['n']}]** {c['source']}{loc} · score {c['score']} · {mark}\n> {c['snippet']}…")
        return "\n\n".join(lines)

    # ---------- handlers ----------
    @friendly
    def on_load():
        if not svc.list_notebooks():
            svc.create_notebook("My first notebook")
        dd = nb_update()
        return (dd, *panel(dd["value"]))

    @friendly
    def create_nb(name):
        nb = svc.create_notebook(name)
        gr.Info(f"Created notebook '{nb['name']}'.")
        return (nb_update(nb["id"]), "", *panel(nb["id"]))

    @friendly
    def rename_nb(nb_id, name):
        svc.rename_notebook(nb_id, name)
        gr.Info("Notebook renamed.")
        return nb_update(nb_id), ""

    @friendly
    def delete_nb(nb_id):
        if not nb_id:
            raise gr.Error("Select a notebook first.")
        svc.delete_notebook(nb_id)
        if not svc.list_notebooks():
            svc.create_notebook("My first notebook")
        dd = nb_update()
        gr.Info("Notebook deleted.")
        return (dd, *panel(dd["value"]))

    @friendly
    def switch_nb(nb_id):
        return panel(nb_id)

    @friendly
    def add_files(nb_id, files):
        if not files:
            raise gr.Error("Choose at least one file to upload.")
        added = []
        for f in files:
            path = getattr(f, "name", f)
            with open(path, "rb") as fh:
                data = fh.read()
            added.append(svc.add_file(nb_id, os.path.basename(path), data)["name"])
        gr.Info(f"Added: {', '.join(added)}")
        return sources_md(nb_id), None

    @friendly
    def add_url(nb_id, url):
        if not (url or "").strip():
            raise gr.Error("Enter a URL first.")
        src = svc.add_url(nb_id, url)
        gr.Info(f"Added web page: {src['name'][:60]}")
        return sources_md(nb_id), ""

    @friendly
    def chat(nb_id, message, mode):
        res = svc.chat(nb_id, message, mode)
        return chat_history(nb_id), "", citations_md(res)

    @friendly
    def generate(nb_id, kind):
        art = svc.generate_artifact(nb_id, kind)
        gr.Info(f"{kind.title()} generated and saved.")
        return art_update(nb_id, art["id"]), art["markdown"], download_file(art)

    @friendly
    def pick_artifact(nb_id, art_id):
        if not art_id:
            return "", None
        art = svc.get_artifact(nb_id, art_id)
        return (art["markdown"], download_file(art)) if art else ("", None)

    # ---------- layout ----------
    with gr.Blocks(title="NotebookLM Clone") as demo:
        gr.Markdown("# 📓 NotebookLM Clone\nUpload sources, chat with citations, and generate reports and quizzes.")
        with gr.Row():
            with gr.Column(scale=1):
                # allow_custom_value: after a server restart/sleep an open tab still holds the old choices; the
                # service validates ids itself, so skip Gradio's strict check instead of erroring.
                nb_dd = gr.Dropdown(label="Notebook", choices=[], interactive=True, allow_custom_value=True)
                name_in = gr.Textbox(label="Notebook name", placeholder="Name for create / rename")
                with gr.Row():
                    create_btn = gr.Button("➕ Create", variant="primary")
                    rename_btn = gr.Button("✏️ Rename")
                    delete_btn = gr.Button("🗑️ Delete", variant="stop")
                gr.Markdown("### Sources")
                files_in = gr.File(label="Upload PDF / PPTX / TXT (added automatically)", file_count="multiple",
                                   file_types=[".pdf", ".pptx", ".txt"])
                url_in = gr.Textbox(label="Web URL", placeholder="https://example.com/article, then press Enter")
                add_url_btn = gr.Button("Add URL")
                src_md = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("💬 Chat"):
                        chatbot = gr.Chatbot(type="messages", height=420, label="Conversation")
                        mode = gr.Radio(["hybrid", "vector"], value="hybrid", label="Retrieval method")
                        msg = gr.Textbox(label="Ask a question about your sources", placeholder="Press Enter to send")
                        send_btn = gr.Button("Send", variant="primary")
                        cite_md = gr.Markdown(label="Citations")
                    with gr.Tab("📄 Artifacts"):
                        with gr.Row():
                            report_btn = gr.Button("Generate Report", variant="primary")
                            quiz_btn = gr.Button("Generate Quiz + Answer Key", variant="primary")
                        art_dd = gr.Dropdown(label="Saved artifacts", choices=[], interactive=True, allow_custom_value=True)
                        art_view = gr.Markdown()
                        art_file = gr.File(label="Download (.md)")

        panel_out = [src_md, chatbot, cite_md, art_dd, art_view, art_file]
        demo.load(on_load, None, [nb_dd, *panel_out])
        nb_dd.input(switch_nb, nb_dd, panel_out)
        create_btn.click(create_nb, name_in, [nb_dd, name_in, *panel_out])
        rename_btn.click(rename_nb, [nb_dd, name_in], [nb_dd, name_in])
        delete_btn.click(delete_nb, nb_dd, [nb_dd, *panel_out])
        files_in.upload(add_files, [nb_dd, files_in], [src_md, files_in])  # ingest as soon as the upload finishes
        for trigger in (url_in.submit, add_url_btn.click):
            trigger(add_url, [nb_dd, url_in], [src_md, url_in])
        for trigger in (msg.submit, send_btn.click):
            trigger(chat, [nb_dd, msg, mode], [chatbot, msg, cite_md])
        report_btn.click(generate, [nb_dd, gr.State("report")], [art_dd, art_view, art_file])
        quiz_btn.click(generate, [nb_dd, gr.State("quiz")], [art_dd, art_view, art_file])
        art_dd.input(pick_artifact, [nb_dd, art_dd], [art_view, art_file])
    return demo


def main():
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    embedder = Embedder()
    threading.Thread(target=lambda: embedder.embed(["warm up"]), daemon=True).start()  # preload model
    svc = NotebookService(get_storage(), embedder, LLMClient())
    build_app(svc).queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
