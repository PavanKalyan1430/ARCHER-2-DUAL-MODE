import redis
import json
import hashlib
import logging
from app.core.config import settings

logger = logging.getLogger("A.R.C.H.E.R.RedisCache")

class RedisCache:
    def __init__(self):
        try:
            self.client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.client.ping()
            self.enabled = True
            logger.info("🔌 Connected to Redis cache service successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Redis is not available: {e}. Running without query cache.")
            self.enabled = False

    def _generate_key(self, query: str, doc_id: str = None, search_mode: str = "quick") -> str:
        """Generate a deterministic cache key using md5 hash."""
        raw_str = f"q:{query.strip().lower()}:d:{doc_id or 'all'}:m:{search_mode}"
        return f"archer:cache:{hashlib.md5(raw_str.encode('utf-8')).hexdigest()}"

    def get(self, query: str, doc_id: str = None, search_mode: str = "quick") -> dict:
        """Retrieve cached response if it exists."""
        if not self.enabled:
            return None
        try:
            key = self._generate_key(query, doc_id, search_mode)
            cached_data = self.client.get(key)
            if cached_data:
                logger.info(f"⚡ Redis Cache Hit for query: '{query}'")
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Redis cache retrieve failed: {e}")
        return None

    def set(self, query: str, response_data: dict, doc_id: str = None, search_mode: str = "quick", ttl: int = 3600):
        """Cache response data in Redis with a Time To Live (TTL) in seconds."""
        if not self.enabled:
            return
        try:
            key = self._generate_key(query, doc_id, search_mode)
            self.client.setex(key, ttl, json.dumps(response_data))
            logger.info(f"💾 Query cached in Redis: '{query}'")
        except Exception as e:
            logger.error(f"Redis cache set failed: {e}")
