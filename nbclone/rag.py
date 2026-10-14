"""RAG chat: retrieve -> build grounded prompt -> generate -> attach citations."""
import re
import time

SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the numbered sources below. "
    "Cite every claim with its source number in square brackets, like [1] or [2][3]. "
    "If the sources do not contain the answer, say you could not find it in the notebook's sources. "
    "Be concise and accurate."
)


def build_prompt(question: str, chunks: list, history: list) -> str:
    sources = "\n\n".join(f"[{i}] ({c['source']}{', ' + c['label'] if c.get('label') else ''})\n{c['text']}"
                          for i, c in enumerate(chunks, 1))
    convo = ""
    if history:
        convo = "Recent conversation:\n" + "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-4:]) + "\n\n"
    return f"{SYSTEM}\n\nSOURCES:\n{sources}\n\n{convo}QUESTION: {question}\n\nANSWER (with [n] citations):"


def make_citations(chunks: list, answer: str) -> list:
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return [{"n": i, "source": c["source"], "label": c.get("label"), "snippet": c["text"][:220].replace("\n", " "),
             "score": c["score"], "cited": i in used} for i, c in enumerate(chunks, 1)]


def format_sources_footer(citations: list) -> str:
    shown = [c for c in citations if c["cited"]] or citations
    lines = [f"- [{c['n']}] {c['source']}{' (' + c['label'] + ')' if c['label'] else ''}" for c in shown]
    return "\n\n**Sources:**\n" + "\n".join(lines)


def answer(question: str, index, llm, mode: str = "hybrid", k: int = 5, history: list | None = None) -> dict:
    t0 = time.perf_counter()
    chunks = index.search(question, k=k, mode=mode)
    t_retrieve = time.perf_counter() - t0
    if not chunks:
        return {"answer": "This notebook has no sources yet. Add a PDF, PPTX, TXT file or URL first.",
                "citations": [], "chunks": [], "retrieval_time": t_retrieve, "llm_time": 0.0}
    t1 = time.perf_counter()
    text = llm.complete(build_prompt(question, chunks, history or []))
    t_llm = time.perf_counter() - t1
    return {"answer": text, "citations": make_citations(chunks, text), "chunks": chunks,
            "retrieval_time": t_retrieve, "llm_time": t_llm}
