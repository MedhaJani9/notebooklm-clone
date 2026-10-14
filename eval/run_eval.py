"""RAG evaluation: vector search vs hybrid (BM25 + vector, RRF) on a fixed question set.

Usage (needs GROQ_API_KEY, internet for the embedding model):
    python eval/run_eval.py                      # uses eval/docs + eval/questions.json
    python eval/run_eval.py --no-judge --sleep 0 # skip LLM-judge scoring

Writes docs/RAG_EVALUATION.md (table, retrieved chunks, timings, conclusions) and eval/results.json.
"""
import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from nbclone import rag  # noqa: E402
from nbclone.embeddings import Embedder  # noqa: E402
from nbclone.llm import LLMClient  # noqa: E402
from nbclone.service import NotebookService  # noqa: E402
from nbclone.storage.local import LocalStorage  # noqa: E402

MODES = ["vector", "hybrid"]
JUDGE_PROMPT = """You are grading a RAG answer. Score 1-5 for each criterion:
relevance (answers the question), groundedness (supported by the sources, no invention).
Reply with ONLY JSON: {{"relevance": n, "groundedness": n, "note": "<=15 words"}}

QUESTION: {q}

SOURCES:
{sources}

ANSWER:
{a}"""


def judge(llm, q, chunks, answer):
    src = "\n\n".join(f"[{i}] {c['text'][:500]}" for i, c in enumerate(chunks, 1))
    raw = llm.complete(JUDGE_PROMPT.format(q=q, sources=src, a=answer), temperature=0)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0))
        return float(d["relevance"]), float(d["groundedness"]), str(d.get("note", ""))
    except Exception:
        return None, None, "judge parse error"


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="eval/docs")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--out", default="docs/RAG_EVALUATION.md")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=6.0, help="pause between LLM calls (free-tier TPM)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    args = ap.parse_args()

    embedder = Embedder()
    llm = LLMClient()
    judge_llm = LLMClient(args.judge_model)
    svc = NotebookService(LocalStorage(tempfile.mkdtemp()), embedder, llm)
    nb = svc.create_notebook("eval")["id"]

    t0 = time.perf_counter()
    for fn in sorted(os.listdir(args.docs)):
        with open(os.path.join(args.docs, fn), "rb") as f:
            src = svc.add_file(nb, fn, f.read())
        print(f"ingested {fn}: {src['chunks']} chunks")
    ingest_s = time.perf_counter() - t0
    total_chunks = sum(s["chunks"] for s in svc.list_sources(nb))
    index = svc.index(nb)
    index.search("warm up", 1, "vector")

    questions = json.load(open(args.questions))
    rows = []
    for item in questions:
        for mode in MODES:
            # retrieval latency: median of 5 runs for stability
            times = []
            for _ in range(5):
                t = time.perf_counter()
                chunks = index.search(item["q"], args.k, mode)
                times.append(time.perf_counter() - t)
            res = rag.answer(item["q"], index, llm, mode=mode, k=args.k)
            time.sleep(args.sleep)
            hits = [any(kw.lower() in c["text"].lower() for kw in item["keywords"]) for c in chunks]
            kw_cov = sum(any(kw.lower() in c["text"].lower() for c in chunks) for kw in item["keywords"]) / len(item["keywords"])
            rel = grd = None
            note = ""
            if not args.no_judge:
                rel, grd, note = judge(judge_llm, item["q"], chunks, res["answer"])
                time.sleep(args.sleep)
            rows.append({"q": item["q"], "mode": mode, "retrieval_ms": statistics.median(times) * 1000,
                         "llm_s": res["llm_time"], "precision": sum(hits) / len(hits), "kw_coverage": kw_cov,
                         "top1_hit": hits[0], "relevance": rel, "groundedness": grd, "note": note,
                         "answer": res["answer"],
                         "chunks": [{"source": c["source"], "label": c["label"], "score": c["score"],
                                     "text": c["text"][:160].replace("\n", " ")} for c in chunks]})
            print(f"{mode:7s} | {item['q'][:50]:50s} | P@{args.k}={rows[-1]['precision']:.2f} "
                  f"ret={rows[-1]['retrieval_ms']:.0f}ms rel={rel} grd={grd}")

    json.dump(rows, open("eval/results.json", "w"), indent=2)
    write_report(rows, args, total_chunks, ingest_s, embedder.model_name, llm.model)
    print(f"\nWrote {args.out} and eval/results.json")


def avg(rows, mode, key):
    vals = [r[key] for r in rows if r["mode"] == mode and r[key] is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def write_report(rows, a, total_chunks, ingest_s, emb, model):
    L = ["# RAG Evaluation: Vector Search vs Hybrid Search", "",
         f"- Corpus: `{a.docs}` ({total_chunks} chunks, ingested in {ingest_s:.1f}s), chunk size/overlap from `nbclone/config.py`",
         f"- Embeddings: `{emb}` · Answer LLM: `{model}` · Judge: `{a.judge_model}` · top-k = {a.k}",
         "- **vector**: cosine similarity over ChromaDB. **hybrid**: vector + BM25 merged with Reciprocal Rank Fusion (k=60).",
         "- *Precision@k* = share of retrieved chunks containing an expected keyword; *Keyword coverage* = share of expected "
         "keywords found anywhere in retrieved chunks; *Relevance/Groundedness* = LLM-judge scores (1-5).", "",
         "## Summary", "", "| Method | Precision@k | Keyword coverage | Top-1 hit | Retrieval ms | LLM s | Relevance | Groundedness |",
         "|---|---|---|---|---|---|---|---|"]
    for m in MODES:
        top1 = sum(r["top1_hit"] for r in rows if r["mode"] == m) / max(1, sum(r["mode"] == m for r in rows))
        L.append(f"| {m} | {avg(rows, m, 'precision'):.2f} | {avg(rows, m, 'kw_coverage'):.2f} | {top1:.2f} | "
                 f"{avg(rows, m, 'retrieval_ms'):.1f} | {avg(rows, m, 'llm_s'):.1f} | {avg(rows, m, 'relevance'):.2f} | "
                 f"{avg(rows, m, 'groundedness'):.2f} |")
    L += ["", "## Per-question results", "",
          "| Question | Method | Retrieved | Precision@k | Retrieval ms | LLM s | Relevance | Groundedness | Judge note |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['q']} | {r['mode']} | {len(r['chunks'])} | {r['precision']:.2f} | {r['retrieval_ms']:.1f} | "
                 f"{r['llm_s']:.1f} | {r['relevance']} | {r['groundedness']} | {r['note']} |")
    L += ["", "## Retrieved chunks and answers", ""]
    for r in rows:
        L += [f"### {r['q']} — *{r['mode']}*", ""]
        for i, c in enumerate(r["chunks"], 1):
            L.append(f"{i}. `{c['source']}` {c['label'] or ''} (score {c['score']}): {c['text']}…")
        L += ["", f"**Answer:** {r['answer']}", ""]
    v_p, h_p = avg(rows, "vector", "precision"), avg(rows, "hybrid", "precision")
    v_t, h_t = avg(rows, "vector", "retrieval_ms"), avg(rows, "hybrid", "retrieval_ms")
    v_q = avg(rows, "vector", "relevance") + avg(rows, "vector", "groundedness")
    h_q = avg(rows, "hybrid", "relevance") + avg(rows, "hybrid", "groundedness")
    L += ["## Conclusions (auto-generated from the numbers above — edit with your own analysis)", "",
          f"- Retrieval precision: hybrid {h_p:.2f} vs vector {v_p:.2f}.",
          f"- Retrieval latency: hybrid {h_t:.1f} ms vs vector {v_t:.1f} ms (hybrid adds a BM25 pass; both are negligible next to the LLM call).",
          f"- Answer quality (judge, relevance+groundedness): hybrid {h_q:.2f} vs vector {v_q:.2f} out of 10.",
          "- **Selected default:** " + ("hybrid" if (h_p >= v_p or h_q >= v_q) else "vector")
          + " search; both remain selectable in the UI."]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
