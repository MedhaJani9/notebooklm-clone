"""Central configuration, all overridable through environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()  # before reading any setting; real env vars (Render) still take precedence

DATA_DIR = os.getenv("DATA_DIR", "data")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()  # "local" | "firestore"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "8"))  # small batches keep peak RAM under 512 MB
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "1"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ARTIFACT_CONTEXT_CHARS = int(os.getenv("ARTIFACT_CONTEXT_CHARS", "9000"))  # keeps prompts inside free-tier TPM
