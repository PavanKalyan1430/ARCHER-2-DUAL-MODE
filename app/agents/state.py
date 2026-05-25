from typing import List, TypedDict, Optional
from app.models.schemas import ChunkSchema

class AgentState(TypedDict):
    """
    The shared memory notepad passed between LangGraph agents.
    Every time an agent runs, it updates this state and passes it to the next agent.
    """
    user_query: str
    session_id: str
    rewritten_query: str
    retrieved_context: List[ChunkSchema]
    is_hallucinated: bool
    is_relevant: bool
    final_answer: str
    attempt_count: int
    doc_id: Optional[str]
    search_mode: Optional[str]

