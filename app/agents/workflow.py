from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes import AgentNodes

class ArcherWorkflow:
    def __init__(self, nodes: AgentNodes):
        self.nodes = nodes
        self.workflow = StateGraph(AgentState)
        self._build_graph()

    def _build_graph(self):
        # 1. Add Nodes (Registering the Agents)
        self.workflow.add_node("rewriter", self.nodes.rewrite_query)
        self.workflow.add_node("retriever", self.nodes.retrieve_context)
        self.workflow.add_node("grader", self.nodes.grade_documents)
        self.workflow.add_node("generator", self.nodes.generate_answer)
        self.workflow.add_node("hallucination_checker", self.nodes.check_hallucination)

        # 2. Define the Entry Point
        self.workflow.set_entry_point("rewriter")

        # 3. Define Standard Edges (Straight lines)
        self.workflow.add_edge("rewriter", "retriever")
        self.workflow.add_edge("retriever", "grader")

        # 4. Define Conditional Edges (The Loops / Decision Making)
        self.workflow.add_conditional_edges(
            "grader",
            self._route_after_grading,
            {
                "generate": "generator",
                "rewrite": "rewriter" # Loop back if context is irrelevant
            }
        )

        self.workflow.add_edge("generator", "hallucination_checker")

        self.workflow.add_conditional_edges(
            "hallucination_checker",
            self._route_after_hallucination_check,
            {
                "end": END,
                "regenerate": "generator" # Loop back if it hallucinated
            }
        )

        # 5. Compile the state machine
        self.app = self.workflow.compile()

    def _route_after_grading(self, state: AgentState) -> str:
        """Decides where to go after the Grader checks the documents."""
        # If we failed to find relevant docs 3 times, give up to prevent infinite loops
        if state.get("attempt_count", 0) >= 3:
            return "generate"
        if state.get("is_relevant", False):
            return "generate"
        return "rewrite"

    def _route_after_hallucination_check(self, state: AgentState) -> str:
        """Decides where to go after checking the final answer for lies."""
        if state.get("attempt_count", 0) >= 3:
            return "end"
        if state.get("is_hallucinated", False):
            return "regenerate"
        return "end"

    def run(self, query: str, session_id: str, search_mode: str = "quick", doc_id: str = None) -> dict:
        """Executes the workflow and returns the final state."""
        initial_state = {
            "user_query": query,
            "session_id": session_id,
            "search_mode": search_mode,
            "doc_id": doc_id,
            "rewritten_query": "",
            "retrieved_context": [],
            "is_hallucinated": False,
            "is_relevant": False,
            "final_answer": "",
            "attempt_count": 0
        }
        
        return self.app.invoke(initial_state)
