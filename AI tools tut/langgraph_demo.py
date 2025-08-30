"""
This module demonstrates the use of LangGraph.

Key differences between LangChain and LangGraph:
- LangChain is primarily designed for building language model-powered applications using chains and agents, focusing on sequential and branching workflows.
- LangGraph introduces a graph-based approach, allowing for more flexible, non-linear workflows and complex state management between nodes.

Use cases:
# LangChain:
# - Conversational agents
# - Question answering systems
# - Sequential data processing pipelines

# LangGraph:
# - Complex multi-step reasoning tasks
# - Dynamic workflows with conditional logic
# - Applications requiring stateful interactions and flexible execution paths
"""

# langgraph_agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict
from euri_llm import generate_completion

# ✅ Step 1: Define a state schema using TypedDict
class GraphState(TypedDict):
    input: str
    research: str
    summary: str

# ✅ Step 2: Define node functions
def researcher(state: GraphState) -> dict:
    question = state["input"]
    print(question)
    response = generate_completion(question)
    return {"research": response}

def summarizer(state: GraphState) -> dict:
    research = state["research"]
    summary = generate_completion(research)
    return {"summary": summary}

# ✅ Step 3: Define and compile LangGraph
graph = StateGraph(GraphState)  # schema required here!
graph.add_node("researcher", researcher)
graph.add_node("summarizer", summarizer)
graph.set_entry_point("researcher")
graph.add_edge("researcher", "summarizer")
graph.set_finish_point("summarizer")

compiled_graph = graph.compile()

# ✅ Step 4: Run the graph
result = compiled_graph.invoke({"input": "Impact of AI in healthcare"})
print("\n✅ Final Summary:\n", result["summary"])
