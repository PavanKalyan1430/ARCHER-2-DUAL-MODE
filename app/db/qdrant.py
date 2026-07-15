from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app.core.config import settings
import logging

logger = logging.getLogger("A.R.C.H.E.R.Qdrant")

# bge-small-en-v1.5 → 384-dim dense vectors (fast, accurate)
DENSE_VECTOR_SIZE = 384

class QdrantManager:
    def __init__(self, collection_name: str = "archer_documents"):
        self.collection_name = collection_name
        try:
            self.client = QdrantClient(url=settings.QDRANT_URL, timeout=60.0)
            # Test connection to ensure server is actually reachable
            self.client.get_collections()
            logger.info("🔌 Connected to Qdrant Docker service successfully.")
        except Exception as e:
            logger.warning(
                f"⚠️ Could not reach Qdrant Docker service at {settings.QDRANT_URL}: {e}. "
                "Falling back to local on-disk Qdrant storage for high resilience!"
            )
            import os
            os.makedirs("temp_uploads/local_qdrant", exist_ok=True)
            self.client = QdrantClient(path="temp_uploads/local_qdrant")
        
        self._ensure_collection_exists()

        # enable_hybrid=True activates Qdrant's native RRF fusion
        # (dense bge-small + sparse SPLADE_PP_en_v1 — already cached locally)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="prithivida/Splade_PP_en_v1",
            sparse_vector_name="text-sparse"
        )

    def _ensure_collection_exists(self):
        """Creates or auto-repairs the collection if the vector size is wrong."""
        collections = self.client.get_collections().collections
        existing = next((c for c in collections if c.name == self.collection_name), None)

        if existing:
            info = self.client.get_collection(self.collection_name)
            try:
                current_size = info.config.params.vectors.size
            except AttributeError:
                current_size = DENSE_VECTOR_SIZE  # Already correct format

            has_sparse = False
            try:
                if getattr(info.config.params, "sparse_vectors", None) and "text-sparse" in info.config.params.sparse_vectors:
                    has_sparse = True
            except Exception:
                pass

            if current_size != DENSE_VECTOR_SIZE or not has_sparse:
                error_msg = (
                    f"Collection configuration mismatch detected for '{self.collection_name}'. "
                    f"Expected vector size: {DENSE_VECTOR_SIZE}, got: {current_size}. "
                    f"Expected sparse vectors enabled: True, got: {has_sparse}. "
                    "To prevent catastrophic data loss, A.R.C.H.E.R will not silently delete this collection. "
                    "Please manually delete or migrate the collection in Qdrant before restarting."
                )
                logger.critical(error_msg)
                raise ValueError(error_msg)

        if not existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE
                ),
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            logger.info(
                f"Created hybrid collection '{self.collection_name}' "
                f"(dense={DENSE_VECTOR_SIZE}d + sparse SPLADE)."
            )

    def get_store(self) -> QdrantVectorStore:
        return self.vector_store
