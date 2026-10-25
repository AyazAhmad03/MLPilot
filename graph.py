from langgraph.graph import StateGraph, START, END

from core.state import GraphState
from core.gates import data_quality_gate, performance_gate

from agents.data import data_upload_agent, data_analysis_agent
from agents.problem_detection import problem_detection_agent
from agents.preparation import data_preparation_agent
from agents.cleaning import data_cleaning_agent
from agents.model_selection import model_selection_agent
from agents.experimentation import model_experimentation_agent
from agents.training import training_agent
from agents.tuning import tuning_agent
from agents.evaluation import evaluation_agent
from agents.reporting import report_agent
from agents.insight import insight_agent

def build_graph():
    graph = StateGraph(GraphState)

    # ---- register all nodes (name -> function) ----
    nodes = {
        "data_upload": data_upload_agent,
        "data_analysis": data_analysis_agent,
        "data_cleaning": data_cleaning_agent,
        "problem_detection": problem_detection_agent,
        "data_preparation": data_preparation_agent,
        "model_selection": model_selection_agent,
        "model_experimentation": model_experimentation_agent,
        "training": training_agent,
        "tuning": tuning_agent,
        "evaluation": evaluation_agent,
        "report": report_agent,
        "insight": insight_agent, 
    }
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # ---- straight edges ----
    graph.add_edge(START, "data_upload")
    graph.add_edge("data_upload", "data_analysis")

    # ---- LOOP 1: the quality gate ----
    # gate returns "needs_cleaning" -> go clean. returns "ready" -> continue.
    graph.add_conditional_edges("data_analysis", data_quality_gate, {
        "needs_cleaning": "data_cleaning",
        "ready": "problem_detection",
    })
    graph.add_edge("data_cleaning", "data_analysis")   # <-- the loop back

    graph.add_edge("problem_detection", "data_preparation")
    graph.add_edge("data_preparation", "model_selection")
    graph.add_edge("model_selection", "model_experimentation")
    graph.add_edge("model_experimentation", "training")
    graph.add_edge("training", "evaluation")

    # ---- LOOP 2: the performance gate ----
    # gate returns "tune" -> tune the model. returns "report" -> finish.
    graph.add_conditional_edges("evaluation", performance_gate, {
        "tune": "tuning",
        "report": "insight",
    })
    graph.add_edge("tuning", "training")               # <-- the loop back
    graph.add_edge("insight", "report")  
    graph.add_edge("report", END)

    return graph.compile()