"""Dense semantic embeddings via sentence-transformers (optional dependency).

Install with ``pip install 'investor-intel[semantic]'`` or
``pip install sentence-transformers``. The default local profile stays
dependency-free TF-IDF; this embedder is selected with
``II_EMBEDDINGS_PROVIDER=sentence-transformers``.

Vectors are dense ``list[float]`` scored by the dense in-memory vector
store, so retrieval still runs fully locally on CPU with no API keys -
the model itself is downloaded once from the Hugging Face hub and cached.
"""
from __future__ import annotations

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Embedder protocol implementation backed by a sentence-transformers model."""

    vector_kind = "dense"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with "
                "'pip install sentence-transformers' (or the 'semantic' extra), "
                "or leave II_EMBEDDINGS_PROVIDER=local for the default TF-IDF retrieval."
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        # Pretrained dense models need no corpus fitting; kept for the protocol.
        return None

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text)]
