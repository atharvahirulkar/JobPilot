from typing import List, Iterable
import os

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


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
        if self.openai_key and OpenAI is not None:
            try:
                client = OpenAI(api_key=self.openai_key)
                resp = client.embeddings.create(model=self.openai_model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception:
                # Fall through to local sentence-transformers on any OpenAI failure
                # (quota exceeded, network, invalid key, etc.)
                pass
        # fallback to sentence-transformers — always return plain Python floats
        self._ensure_hf()
        vectors = self._hf.encode(texts, convert_to_numpy=True)
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [list(v) if hasattr(v, "tolist") else v for v in vectors]


if __name__ == "__main__":
    ec = EmbeddingClient()
    print(ec.embed_texts(["Machine learning", "Data engineering"]))
