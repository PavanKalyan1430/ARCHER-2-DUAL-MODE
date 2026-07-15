from typing import List
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings as LlamaIndexSettings

class EmbeddingService:
    def __init__(self):
        # 1. Initialize the embedding model using FastEmbed
        # FastEmbed is written in Rust and uses the ONNX Runtime for extreme CPU optimization.
        # We use bge-small-en-v1.5 which is incredibly fast on CPU and still highly accurate.
        import os
        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.model_cache/fastembed"))
        os.makedirs(cache_path, exist_ok=True)
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=cache_path)
        
        # 2. Register it globally in LlamaIndex
        LlamaIndexSettings.embed_model = self.embed_model

    def embed_text(self, text: str) -> List[float]:
        """Manually embed a single string (useful for chat queries)."""
        return self.embed_model.get_text_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text chunks simultaneously for faster processing."""
        return self.embed_model.get_text_embedding_batch(texts)
        
    def get_model(self) -> FastEmbedEmbedding:
        """Returns the raw LlamaIndex-compatible embedding object."""
        return self.embed_model
