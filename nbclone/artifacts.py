"""Artifact generation (report, quiz + answer key) grounded in notebook chunks."""
from . import config


def sample_chunks(chunks: list, max_chars: int | None = None) -> list:
    """Pick chunks evenly spread across the notebook so the artifact covers all sources
    while staying inside the model's token budget."""
    max_chars = max_chars or config.ARTIFACT_CONTEXT_CHARS
    if not chunks:
        return []
    avg = max(1, sum(len(c["text"]) for c in chunks) // len(chunks))
    n = max(1, min(len(chunks), max_chars // avg))
    step = len(chunks) / n
    return [chunks[int(i * step)] for i in range(n)]


def _context(chunks: list) -> str:
    return "\n\n".join(f"[{c['source']}{', ' + c['label'] if c.get('label') else ''}]\n{c['text']}" for c in chunks)


REPORT_PROMPT = """Write a well-structured study report in Markdown based ONLY on the source excerpts below.
Use this structure: a '# ' title, '## Executive Summary', 3-5 '## ' thematic sections with bullet points,
'## Key Takeaways', and '## Sources Referenced' (list the source names). Do not invent facts.

SOURCE EXCERPTS:
{context}
"""

QUIZ_PROMPT = """Create a quiz in Markdown based ONLY on the source excerpts below.
Format: '# Quiz' title, then 8 numbered questions: 6 multiple-choice (options A-D) and 2 short-answer.
Then a final section titled exactly '## Answer Key' giving the correct answer for every question
with a one-sentence explanation. Do not invent facts.

SOURCE EXCERPTS:
{context}
"""


def generate(kind: str, chunks: list, llm) -> str:
    if not chunks:
        raise ValueError("Add at least one source before generating an artifact.")
    ctx = _context(sample_chunks(chunks))
    if kind == "report":
        return llm.complete(REPORT_PROMPT.format(context=ctx), temperature=0.3)
    if kind == "quiz":
        md = llm.complete(QUIZ_PROMPT.format(context=ctx), temperature=0.3)
        if "answer key" not in md.lower():  # guarantee the required answer key
            key = llm.complete("Below is a quiz. Write only a '## Answer Key' section with the correct answer "
                               "and a one-sentence explanation for each question.\n\n" + md, temperature=0.0)
            md = md.rstrip() + "\n\n" + key
        return md
    raise ValueError(f"Unknown artifact type '{kind}'.")
