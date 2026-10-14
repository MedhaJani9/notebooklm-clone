# 📓 NotebookLM Clone

A NotebookLM-style research assistant. Create multiple notebooks, add PDF / PPTX / TXT files or web URLs, chat with your
sources using **Retrieval-Augmented Generation with citations**, and generate a **report** and a **quiz with answer key**
(downloadable `.md`). Built with Gradio, ChromaDB, BM25, fastembed, Groq (LangChain) and Firestore.

- **Live app:** `<paste your Render URL>`
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · **RAG evaluation:** [docs/RAG_EVALUATION.md](docs/RAG_EVALUATION.md)

## Features

| Area | What it does |
|---|---|
| Notebooks | Create, rename, delete, switch. Sources, chats and artifacts are stored per notebook id. |
| Ingestion | PDF (per page), PPTX (per slide, incl. tables/notes), TXT, single-page URL → extract → chunk → embed → index. |
| RAG chat | Top-5 retrieval (**vector** or **hybrid BM25+vector/RRF**, selectable), grounded answer with `[n]` citations, source/page/slide + snippets + timings shown. History is saved. |
| Artifacts | Report and Quiz (+ Answer Key) generated from notebook content, viewable in-app and downloadable as `.md`. |
| Errors | Friendly messages for bad files, empty input, blocked URLs, rate limits and missing keys. |

## Run locally

```bash
python -m venv venv && source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                                    # then put your GROQ_API_KEY in .env
python app.py                                           # http://localhost:7860
pytest -q                                               # 32 tests, no network/API key needed
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | – (required) | Groq API key (secret) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Chat/artifact model |
| `STORAGE_BACKEND` | `local` | `local` (disk, `./data`) or `firestore` |
| `GOOGLE_CREDENTIALS_JSON` | – | Service-account JSON (one line) for Firestore (secret) |
| `DATA_DIR` | `data` | Local storage root |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | fastembed model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` | 800 / 100 / 5 | RAG parameters |
| `PORT` | 7860 (Render sets it) | HTTP port |

## Project structure

```
app.py                      Gradio UI
nbclone/  service.py        orchestration facade
          ingestion.py      extract + chunk (PDF/PPTX/TXT/URL)
          embeddings.py     fastembed wrapper
          retrieval.py      Chroma vector + BM25 hybrid (RRF)
          rag.py            grounded prompt + citations
          artifacts.py      report / quiz generation
          llm.py            Groq client, retries, friendly errors
          storage/          Storage interface, local + Firestore backends
eval/     run_eval.py, questions.json, docs/Python.pdf
tests/    pytest suite (both storage backends, ingestion, retrieval, service, UI build)
docs/     ARCHITECTURE.md, RAG_EVALUATION.md
Dockerfile · render.yaml · .github/workflows/deploy.yml
```

## How data is stored

Every notebook has a unique id. Local mode: `data/notebooks/<id>/` holds `raw/`, `extracted/`, `chunks.json` (text + embeddings),
`messages.json`, `artifacts/*.md`. Firestore mode: `notebooks/{id}/{sources,chunks,messages,artifacts}`.
The in-memory vector index is rebuilt from stored chunks after a restart.

**Limitations:** Render's filesystem is ephemeral, so deploy with `STORAGE_BACKEND=firestore` for persistence. Firestore stores chunks, chats and
artifacts but not the original uploaded binaries (the extracted text is kept in chunks). Scanned/image-only PDFs have no text layer and are rejected with a message.
The free Render instance sleeps when idle (first request is slow) and has 512 MB RAM; use the Starter plan if it runs out of memory.
Groq's free tier is rate-limited; the app retries with back-off and shows a friendly message.

## Deploy (Render + Firestore + GitHub Actions)

1. **Firestore:** Google Cloud console → create a project → Firestore → *Create database*, Native mode, ID `(default)`.
   IAM → Service Accounts → create one with role **Cloud Datastore User** → Keys → create JSON key. Collapse the JSON to one line.
2. **Render:** New → *Blueprint* (or Web Service → Docker) from this repo (`render.yaml` is included). Set env vars `GROQ_API_KEY` and
   `GOOGLE_CREDENTIALS_JSON`. Auto-Deploy is **off** (GitHub Actions deploys). Settings → *Deploy Hook* → copy the URL.
3. **GitHub:** repo → Settings → Secrets and variables → Actions → add `RENDER_DEPLOY_HOOK_URL` (the hook URL) and `APP_URL` (your `https://….onrender.com`).
4. `git push origin main` → Actions runs `pytest`, then triggers the Render deploy and waits for the app to answer.

No secrets are committed: `.env` and key files are git-ignored; keys live in GitHub Secrets and Render env vars.

## Usage

1. Create (or pick) a notebook → add sources (upload files or paste a URL).
2. **Chat** tab: choose *hybrid* or *vector* retrieval, ask a question, read the answer; the citations panel shows each retrieved chunk, its score and timings.
3. **Artifacts** tab: *Generate Report* / *Generate Quiz + Answer Key*, view it, download the `.md`.

## RAG evaluation

```bash
python eval/run_eval.py        # ingests eval/docs, compares vector vs hybrid, writes docs/RAG_EVALUATION.md
```
