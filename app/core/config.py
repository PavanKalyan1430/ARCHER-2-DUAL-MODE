from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    """
    Core Application Settings for A.R.C.H.E.R.
    Reads directly from the .env file.
    """
    
    # ==========================================
    # LLM Provider Configurations
    # ==========================================
    # 3 Groq keys for round-robin load balancing (avoids rate limits)
    GROQ_API_KEY_1: str = Field(..., description="Groq Key 1")
    GROQ_API_KEY_2: str = Field(..., description="Groq Key 2")
    GROQ_API_KEY_3: str = Field(..., description="Groq Key 3")
    
    # ==========================================
    # Default Models
    # ==========================================
    FAST_LLM_MODEL: str = Field(default="meta-llama/llama-3.2-3b-instruct:free")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-base-en")

    # ==========================================
    # Vector Database Configuration
    # ==========================================
    QDRANT_URL: str = Field(..., description="Qdrant connection string")

    # ==========================================
    # Redis Cache Configuration
    # ==========================================
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # ==========================================
    # Neo4j Graph Database Configuration
    # ==========================================
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="archer_graph_pass", description="Neo4j password")

    # ==========================================
    # Tavily Search API Configuration
    # ==========================================
    TAVILY_API_KEY: Optional[str] = Field(default=None, description="Tavily Web Search Key")

    # This tells Python to look for the .env file in the root folder
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Create a global settings object that we will import everywhere
settings = Settings()
