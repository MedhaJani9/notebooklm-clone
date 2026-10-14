"""Thin Groq/LangChain wrapper with retry + friendly errors."""
import logging
import os
import time

from . import config

log = logging.getLogger(__name__)


class LLMError(Exception):
    """User-facing LLM failure."""


class LLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or config.GROQ_MODEL
        self._llm = {}

    def _get(self, temperature: float):
        if temperature not in self._llm:
            if not os.getenv("GROQ_API_KEY"):
                raise LLMError("GROQ_API_KEY is not configured on the server.")
            from langchain_groq import ChatGroq
            kwargs = {}
            if self.model.startswith(("openai/gpt-oss", "qwen/")):
                kwargs["reasoning_format"] = "hidden"  # keep <think> traces out of answers
            if self.model.startswith("openai/gpt-oss"):
                kwargs["reasoning_effort"] = "low"     # saves tokens on rate-limited free tiers
            self._llm[temperature] = ChatGroq(model=self.model, temperature=temperature, **kwargs)
        return self._llm[temperature]

    def complete(self, prompt: str, temperature: float = 0.2, retries: int = 3) -> str:
        for attempt in range(1, retries + 1):
            try:
                return self._get(temperature).invoke(prompt).content.strip()
            except LLMError:
                raise
            except Exception as e:
                msg = str(e).lower()
                if ("429" in msg or "rate" in msg) and attempt < retries:
                    time.sleep(8 * attempt)
                    continue
                log.exception("LLM call failed")
                if "429" in msg or "rate" in msg:
                    raise LLMError("The model is rate-limited right now. Please wait a minute and try again.")
                if "404" in msg or "model_not_found" in msg:
                    raise LLMError(f"Model '{self.model}' is not available for this API key.")
                raise LLMError(f"The language model request failed ({e.__class__.__name__}).")
