from app.agents.state import AgentState
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService
from llama_index.core import PromptTemplate

class AgentNodes:
    """Contains the individual AI Agents that make up the A.R.C.H.E.R. workflow."""
    def __init__(self, llm_service: LLMService, retrieval_service: RetrievalService):
        self.fast_llm = llm_service.get_fast_llm()
        self.smart_llm = llm_service.get_smart_llm()
        self.retrieval = retrieval_service

    def rewrite_query(self, state: AgentState) -> AgentState:
        """Agent 1: Optimizes the user query for better database searching."""
        search_mode = state.get("search_mode", "quick")
        if search_mode == "quick":
            # Flash Mode: skip query rewriting completely for maximum speed
            state["rewritten_query"] = state["user_query"].strip()
            state["attempt_count"] = state.get("attempt_count", 0) + 1
            return state
            
        # Pro Mode: full agentic query rewriting using LLM
        query = state["user_query"].strip()
        prompt = PromptTemplate(
            "Rewrite the following query to be a highly effective search query for a vector database. "
            "Remove conversational filler and focus on keywords.\n"
            "Query: {query}\nRewritten Query:"
        )
        response = self.fast_llm.complete(prompt.format(query=query))
        state["rewritten_query"] = str(response).strip()
        state["attempt_count"] = state.get("attempt_count", 0) + 1
        return state

    def retrieve_context(self, state: AgentState) -> AgentState:
        """Agent 2: Fetches hybrid results from Qdrant."""
        search_mode = state.get("search_mode", "quick")
        # Flash Mode retrieves top_k=3, Pro Mode retrieves top_k=8
        top_k = 3 if search_mode == "quick" else 8
        chunks = self.retrieval.retrieve_hybrid(
            query=state["rewritten_query"], 
            top_k=top_k,
            doc_id=state.get("doc_id")
        )
        
        # Deduplicate and accumulate chunks across loop attempts
        existing_chunks = state.get("retrieved_context") or []
        if not existing_chunks:
            state["retrieved_context"] = chunks
        else:
            existing_ids = {c.chunk_id for c in existing_chunks}
            new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
            state["retrieved_context"] = existing_chunks + new_chunks
            
        return state

    def grade_documents(self, state: AgentState) -> AgentState:
        """Agent 3: Evaluates if the retrieved documents actually answer the question."""
        if not state["retrieved_context"]:
            state["is_relevant"] = False
            return state
            
        search_mode = state.get("search_mode", "quick")
        if search_mode == "quick":
            # Flash Mode: bypass document grading for speed
            state["is_relevant"] = True
            return state
            
        # Pro Mode: full relevance grading via LLM
        context_str = "\n".join([c.text for c in state["retrieved_context"]])
        prompt = PromptTemplate(
            "You are a strict grader evaluating document relevance. Does this context contain the answer to the query?\n"
            "Query: {query}\nContext: {context}\n"
            "Answer strictly 'yes' or 'no'."
        )
        response = self.fast_llm.complete(prompt.format(query=state["user_query"], context=context_str))
        state["is_relevant"] = "yes" in str(response).lower()
        return state

    def generate_answer(self, state: AgentState) -> AgentState:
        """Agent 4: Generates the final answer strictly using the context."""
        context_str = "\n".join([c.text for c in state["retrieved_context"]])
        prompt = PromptTemplate(
            "Answer the query based ONLY on the provided context. If the answer is not in the context, say 'I don't know'.\n"
            "Query: {query}\nContext:\n{context}\nAnswer:"
        )
        # 🌟 USE THE SMART LLM HERE FOR MAXIMUM REASONING IQ 🌟
        response = self.smart_llm.complete(prompt.format(query=state["user_query"], context=context_str))
        state["final_answer"] = str(response).strip()
        return state

    def check_hallucination(self, state: AgentState) -> AgentState:
        """Agent 5: Double-checks that the final answer isn't hallucinated (made up)."""
        search_mode = state.get("search_mode", "quick")
        if search_mode == "quick":
            # Flash Mode: bypass hallucination check for speed
            state["is_hallucinated"] = False
            return state
            
        # Pro Mode: strict factuality check via LLM
        context_str = "\n".join([c.text for c in state["retrieved_context"]])
        prompt = PromptTemplate(
            "Is the following answer factually supported by the context provided?\n"
            "Context: {context}\nAnswer: {answer}\n"
            "Reply strictly 'yes' (it is supported) or 'no' (it is a hallucination)."
        )
        response = self.fast_llm.complete(prompt.format(context=context_str, answer=state["final_answer"]))
        state["is_hallucinated"] = "no" in str(response).lower()
        return state
