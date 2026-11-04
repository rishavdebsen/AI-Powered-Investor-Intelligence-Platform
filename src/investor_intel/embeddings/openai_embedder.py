"""OpenAI-compatible embeddings (also Azure OpenAI via OPENAI_BASE_URL or the
AZURE_OPENAI_* variables). Imported lazily by the factory; the `openai`
package is only required when this embedder is actually selected."""
from __future__ import annotations


class OpenAIEmbedder:
    def __init__(self, api_key: str, base_url: str = "", model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI  # lazy

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    def fit(self, texts: list[str]) -> None:  # stateless
        return None

    def embed(self, text: str) -> dict[str, float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return {str(i): v for i, v in enumerate(resp.data[0].embedding)}
