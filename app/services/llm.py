from itertools import cycle
from llama_index.llms.groq import Groq
from llama_index.core import Settings as LlamaIndexSettings
from app.core.config import settings
import time
import logging
import threading

logger = logging.getLogger("A.R.C.H.E.R.LLM")

# Groq round-robin key pool
GROQ_KEYS = [
    settings.GROQ_API_KEY_1,
    settings.GROQ_API_KEY_2,
    settings.GROQ_API_KEY_3,
]
GROQ_KEY_POOL = cycle(GROQ_KEYS)

# Groq offers both blazing fast and highly intelligent models!
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
GROQ_SMART_MODEL = "llama-3.3-70b-versatile"

def _build_groq(api_key: str, model: str, temperature: float) -> Groq:
    """Helper to instantiate a Groq LLM client."""
    return Groq(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )

class ResilientGroqLLM:
    """
    Round-robin across 3 Groq keys for lightning-fast, rate-limit-proof calls.
    It automatically cycles through the API keys if one gets rate-limited.
    """
    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        # Pre-instantiate Groq clients to leverage HTTP keep-alive connection pooling
        self.clients = []
        for key in GROQ_KEYS:
            if key and key.strip():
                self.clients.append(_build_groq(key, self.model, self.temperature))
        
        # Fallback if no keys in pool
        if not self.clients:
            import os
            fallback_key = os.environ.get("GROQ_API_KEY") or (GROQ_KEYS[0] if GROQ_KEYS else "")
            logger.warning(f"No keys found in pool for model {model}. Using fallback key.")
            self.clients.append(_build_groq(fallback_key, self.model, self.temperature))
            
        self.current_index = 0
        self.lock = threading.Lock()

    def complete(self, prompt: str) -> object:
        num_keys = len(self.clients)
        # Try each pre-instantiated client. We allow 2 full cycles.
        for attempt in range(num_keys * 2): 
            with self.lock:
                client = self.clients[self.current_index]
                # Rotate key index for the next call
                self.current_index = (self.current_index + 1) % num_keys
            try:
                logger.info(f"LLM: Calling pre-instantiated Groq ({self.model}) client...")
                return client.complete(prompt)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower() or "limit" in error_str.lower():
                    logger.warning("Groq key rate-limited, rotating to next key in pool...")
                    time.sleep(0.5) # Brief pause before trying the next key
                    continue
                # If it's not a rate limit error, raise it immediately
                raise e

        # All Groq keys exhausted
        logger.error(f"All Groq keys rate-limited for {self.model}.")
        raise Exception("All Groq API keys are currently rate-limited. Please wait a minute and try again.")


class LLMService:
    def __init__(self):
        # The Fast LLM is used for intermediate agent steps (Grader, Rewriter, Checker)
        self.fast_llm = ResilientGroqLLM(model=GROQ_FAST_MODEL, temperature=0.1)
        
        # The Smart LLM is used for generating the final comprehensive answer
        self.smart_llm = ResilientGroqLLM(model=GROQ_SMART_MODEL, temperature=0.3)

        # Register a single Groq instance globally for LlamaIndex internals
        LlamaIndexSettings.llm = _build_groq(GROQ_KEYS[0], GROQ_FAST_MODEL, 0.1)
        logger.info("LLM Service initialized: Pure 3-key Groq pool for both Fast and Smart agents.")

    def get_fast_llm(self) -> ResilientGroqLLM:
        return self.fast_llm

    def get_smart_llm(self) -> ResilientGroqLLM:
        return self.smart_llm
