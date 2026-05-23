from typing import Iterable, Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from .embeddings import EmbeddingClient


def ingest_texts_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    texts: Iterable[str],
    metadatas: Iterable[Dict[str, Any]],
    ids: Iterable[str] = None,
):
    """Compute embeddings and upsert points into Qdrant collection."""
    emb = EmbeddingClient()
    texts = list(texts)
    metadatas = list(metadatas)
    if ids is None:
        ids = [str(i) for i in range(len(texts))]
    else:
        ids = list(ids)

    vectors = emb.embed_texts(texts)
    points: List[PointStruct] = []
    for _id, vector, meta, text in zip(ids, vectors, metadatas, texts):
        payload = dict(meta or {})
        payload["text"] = text
        points.append(PointStruct(id=_id, vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)


def ensure_text_collection(client: QdrantClient, collection_name: str, vector_size: int):
    from qdrant_client.http.models import Distance, VectorParams

    try:
        collection_names = [c.name for c in client.get_collections().collections]
    except Exception:
        collection_names = []

    if collection_name not in collection_names:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


if __name__ == "__main__":
    from jobpilot.qdrant_setup import init_qdrant
    client = init_qdrant()
    ingest_texts_to_qdrant(client, "jobpilot_resumes", ["Data scientist with 3 years..."], [{"source":"demo"}], ids=["demo-1"]) 
    print("Upsert done")
