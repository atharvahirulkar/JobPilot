import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def init_qdrant(
    client_host: str | None = None,
    api_key: str | None = None,
    collection_name: str = "jobpilot_resumes",
    vector_size: int = 384,
) -> QdrantClient:
    """Initialize a Qdrant client and ensure a collection exists.

    Uses `QDRANT_HOST` and `QDRANT_API_KEY` environment variables if available.
    """
    host = client_host or os.getenv("QDRANT_HOST", "127.0.0.1:6333")
    api_key = api_key or os.getenv("QDRANT_API_KEY")
    # Accept host like '127.0.0.1:6333' or 'http://host:port'
    if not host.startswith("http"):
        url = f"http://{host}"
    else:
        url = host

    client = QdrantClient(url=url, api_key=api_key)

    # Check and create collection with a reasonable default vector size placeholder
    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        collections = []

    if collection_name not in collections:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    return client


if __name__ == "__main__":
    c = init_qdrant()
    print("Collections:", c.get_collections())
