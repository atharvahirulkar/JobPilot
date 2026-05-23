from typing import List, Iterable
import os

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None


class EmbeddingClient:
    """Provides embeddings using OpenAI (if API key present) or Sentence-Transformers fallback."""

    def __init__(self, hf_model: str = "all-MiniLM-L6-v2", openai_model: str = "text-embedding-3-small"):
        self.hf_model_name = hf_model
        self.openai_model = openai_model
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self._hf = None

    @property
    def vector_size(self) -> int:
        if self.openai_key:
            return 1536
        return 384

    def _ensure_hf(self):
        if self._hf is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not available; install requirements.txt")
            self._hf = SentenceTransformer(self.hf_model_name)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if self.openai_key and openai is not None:
            openai.api_key = self.openai_key
            resp = openai.Embedding.create(model=self.openai_model, input=texts)
            return [r["embedding"] for r in resp["data"]]
        # fallback to sentence-transformers — always return plain Python floats
        self._ensure_hf()
        vectors = self._hf.encode(texts, convert_to_numpy=True)
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [list(v) if hasattr(v, "tolist") else v for v in vectors]


if __name__ == "__main__":
    ec = EmbeddingClient()
    print(ec.embed_texts(["Machine learning", "Data engineering"]))
