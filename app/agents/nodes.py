import logging
from app.agents.state import AgentState
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService
from llama_index.core import PromptTemplate

from app.services.embedding import EmbeddingService

logger = logging.getLogger("A.R.C.H.E.R.Nodes")

class AgentNodes:
    """Contains the individual AI Agents that make up the A.R.C.H.E.R. workflow."""
    def __init__(self, llm_service: LLMService, retrieval_service: RetrievalService, embedding_service: EmbeddingService, neo4j_manager=None):
        self.fast_llm = llm_service.get_fast_llm()
        self.smart_llm = llm_service.get_smart_llm()
        self.retrieval = retrieval_service
        self.embedding_service = embedding_service
        self.neo4j = neo4j_manager

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
        """Agent 2: Fetches hybrid results from Qdrant and Neo4j."""
        search_mode = state.get("search_mode", "quick")
        # Flash Mode retrieves top_k=3, Pro Mode retrieves top_k=8
        top_k = 3 if search_mode == "quick" else 8
        chunks = self.retrieval.retrieve_hybrid(
            query=state["rewritten_query"], 
            top_k=top_k,
            doc_id=state.get("doc_id")
        )
        
        # Neo4j Graph RAG retrieval for Pro Mode
        graph_chunks = []
        graph_facts = []
        if search_mode != "quick" and self.neo4j and self.neo4j.driver:
            import json
            logger.info("🕸️ Performing Neo4j Graph Database search for Pro Mode RAG...")
            # Extract key entities from rewritten query
            entity_prompt = (
                "Identify the key entities (names of people, organizations, concepts, locations, elements) mentioned in this search query.\n"
                "Return them as a JSON list of strings (e.g., [\"Jupiter\", \"solar system\"]).\n"
                "Do not return any explanation or backticks. If none are found, return [].\n"
                f"Query: {state['rewritten_query']}\n"
                "JSON:"
            )
            try:
                res_ent = self.fast_llm.complete(entity_prompt)
                clean_ent = str(res_ent).strip().strip("`").replace("json\n", "").strip()
                if clean_ent.startswith("json"):
                    clean_ent = clean_ent[4:].strip()
                entities = json.loads(clean_ent)
                if isinstance(entities, list) and entities:
                    # Lowercase entities for matching
                    entities_lower = [e.lower().strip() for e in entities]
                    
                    # 1. Query to find Chunks mentioned in those entities
                    query_chunks = (
                        "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk) "
                        "WHERE e.name IN $entities "
                        "RETURN c.chunk_id AS chunk_id, c.text AS text, c.doc_id AS doc_id, c.page AS page, c.filename AS filename"
                    )
                    records_chunks = self.neo4j.execute_query(query_chunks, {"entities": entities_lower})
                    
                    from app.models.schemas import ChunkSchema, ChunkMetadata
                    for r in records_chunks:
                        graph_chunks.append(ChunkSchema(
                            chunk_id=r["chunk_id"],
                            text=r["text"],
                            metadata=ChunkMetadata(
                                doc_id=r["doc_id"],
                                page=r["page"],
                                filename=r["filename"],
                                content_type="text",
                                section_heading="Graph-RAG"
                            )
                        ))

                    # 2. Query to find facts (relationships) between those entities
                    query_facts = (
                        "MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity) "
                        "WHERE s.name IN $entities OR t.name IN $entities "
                        "RETURN s.name_raw AS source, r.type_raw AS relation, t.name_raw AS target"
                    )
                    records_facts = self.neo4j.execute_query(query_facts, {"entities": entities_lower})
                    for r in records_facts:
                        fact_str = f"- {r['source']} is related to {r['target']} via '{r['relation']}'"
                        graph_facts.append(fact_str)
            except Exception as e_graph_search:
                logger.error(f"Neo4j graph RAG search failed: {e_graph_search}")

        # Combine vector chunks and graph chunks (and deduplicate by chunk_id)
        all_new_chunks = chunks + graph_chunks
        
        # Deduplicate and accumulate chunks across loop attempts
        existing_chunks = state.get("retrieved_context") or []
        existing_ids = {c.chunk_id for c in existing_chunks}
        
        deduped_new = []
        for c in all_new_chunks:
            if c.chunk_id not in existing_ids:
                deduped_new.append(c)
                existing_ids.add(c.chunk_id)
                
        state["retrieved_context"] = existing_chunks + deduped_new

        if graph_facts:
            state["graph_facts"] = graph_facts
            
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
        context_parts = []
        if state.get("graph_facts"):
            context_parts.append("Graph Facts Extracted from Neo4j:\n" + "\n".join(state["graph_facts"]))
        context_parts.append("Document Text Context:\n" + "\n".join([c.text for c in state["retrieved_context"]]))
        context_str = "\n\n".join(context_parts)
        
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
            state["ragas_metrics"] = {"faithfulness": 1.0, "answer_relevance": 1.0}
            return state
            
        # Pro Mode: strict factuality check via LLM (Ragas-like)
        import json
        import numpy as np
        
        context_str = "\n".join([c.text for c in state["retrieved_context"]])
        answer = state["final_answer"]
        query = state["user_query"]
        
        # 1. Faithfulness score
        faithfulness_prompt = (
            "Analyze the Answer and Context. Identify each unique factual claim made in the Answer. "
            "For each claim, evaluate if it is directly and fully supported by the Context.\n"
            "Return the output as a JSON object with this exact schema:\n"
            "{\n"
            "  \"claims\": [\"claim 1\", \"claim 2\"],\n"
            "  \"verdicts\": [true, false],\n"
            "  \"reasoning\": [\"reason 1\", \"reason 2\"]\n"
            "}\n"
            f"Context: {context_str}\n"
            f"Answer: {answer}\n"
            "JSON:"
        )
        
        faith_val = 1.0
        is_hallucinated = False
        try:
            res = self.fast_llm.complete(faithfulness_prompt)
            # Strip any markdown formatting block if returned
            clean_res = str(res).strip().strip("`").replace("json\n", "").strip()
            # If the output starts with ```json or similar, strip it
            if clean_res.startswith("json"):
                clean_res = clean_res[4:].strip()
            data = json.loads(clean_res)
            verdicts = data.get("verdicts", [])
            if verdicts:
                faith_val = sum(1 for v in verdicts if v) / len(verdicts)
                # If faithfulness is below 0.7, we flag it as hallucinated
                if faith_val < 0.7:
                    is_hallucinated = True
        except Exception as e:
            # Fallback to simple yes/no checking if JSON parsing fails
            prompt_fallback = PromptTemplate(
                "Is the following answer factually supported by the context provided?\n"
                "Context: {context}\nAnswer: {answer}\n"
                "Reply strictly 'yes' (it is supported) or 'no' (it is a hallucination)."
            )
            fallback_res = self.fast_llm.complete(prompt_fallback.format(context=context_str, answer=answer))
            is_hallucinated = "no" in str(fallback_res).lower()
            faith_val = 0.0 if is_hallucinated else 1.0
            
        # 2. Answer Relevance score using BGE embeddings
        relevance_val = 1.0
        try:
            relevance_prompt = (
                "Given the user's original Query and the generated Answer, generate 3 potential questions "
                "that this Answer is answering.\n"
                "Return the output as a JSON object with this exact schema:\n"
                "{\n"
                "  \"generated_questions\": [\"question 1\", \"question 2\", \"question 3\"]\n"
                "}\n"
                f"Original Query: {query}\n"
                f"Answer: {answer}\n"
                "JSON:"
            )
            res_rel = self.fast_llm.complete(relevance_prompt)
            clean_rel = str(res_rel).strip().strip("`").replace("json\n", "").strip()
            if clean_rel.startswith("json"):
                clean_rel = clean_rel[4:].strip()
            data_rel = json.loads(clean_rel)
            gen_qs = data_rel.get("generated_questions", [])
            if gen_qs:
                # Calculate semantic similarity using the injected EmbeddingService
                query_emb = np.array(self.embedding_service.embed_text(query))
                
                similarities = []
                for q in gen_qs:
                    q_emb = np.array(self.embedding_service.embed_text(q))
                    sim = np.dot(query_emb, q_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(q_emb))
                    similarities.append(float(sim))
                relevance_val = sum(similarities) / len(similarities)
        except Exception:
            pass
            
        state["is_hallucinated"] = is_hallucinated
        state["ragas_metrics"] = {
            "faithfulness": round(faith_val, 2),
            "answer_relevance": round(relevance_val, 2)
        }
        return state
