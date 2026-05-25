from typing import List
from llama_index.core import VectorStoreIndex
from app.db.qdrant import QdrantManager
from app.models.schemas import ChunkSchema, ChunkMetadata

class RetrievalService:
    def __init__(self, qdrant_manager: QdrantManager):
        # Connect to our Qdrant instance
        self.vector_store = qdrant_manager.get_store()
        
        # Create an index object that LlamaIndex can query against
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

    def retrieve_hybrid(self, query: str, top_k: int = 5, doc_id: str = None) -> List[ChunkSchema]:
        """
        Dense vector retrieval using BAAI/bge-small-en-v1.5 embeddings.
        Fast and accurate for CPU-bound environments.
        """
        filters = None
        if doc_id:
            from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
            filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(key="doc_id", value=doc_id)
                ]
            )

        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            filters=filters,
        )
        
        # Retrieve the most relevant nodes
        nodes = retriever.retrieve(query)
        
        # Convert the LlamaIndex Nodes back into our strict Pydantic Blueprints
        chunks = []
        for node in nodes:
            metadata = node.metadata
            # Look for stored chunk_id, fall back to node.node_id or custom fallback
            chunk_id = metadata.get("chunk_id", node.node_id or f"{metadata.get('doc_id', 'unknown')}_chunk")
            chunks.append(ChunkSchema(
                chunk_id=chunk_id,
                text=node.text,
                metadata=ChunkMetadata(**metadata) if "content_type" in metadata else metadata # Safe fallback
            ))
            
        return chunks
