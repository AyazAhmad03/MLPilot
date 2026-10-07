from langgraph.graph import StateGraph, START, END

from core.state import GraphState

from agents.data import (
    data_upload_agent,
    data_analysis_agent
)

from agents.preparation import (
    problem_detection_agent,
    data_preparation_agent
)

from agents.model_selection import (
    model_selection_agent,
    model_experimentation_agent
)

from agents.training import training_agent
from agents.evaluation import evaluation_agent
from agents.reporting import report_agent


# ============================================================
# CREATE MLPILOT GRAPH
# ============================================================

def create_graph():

    graph = StateGraph(GraphState)

    # --------------------------------------------------------
    # Add Agents
    # --------------------------------------------------------

    graph.add_node("data_upload", data_upload_agent)
    graph.add_node("data_analysis", data_analysis_agent)
    graph.add_node("problem_detection", problem_detection_agent)
    graph.add_node("data_preparation", data_preparation_agent)
    graph.add_node("model_selection", model_selection_agent)
    graph.add_node("model_experimentation", model_experimentation_agent)
    graph.add_node("training", training_agent)
    graph.add_node("evaluation", evaluation_agent)
    graph.add_node("reporting", report_agent)

    # --------------------------------------------------------
    # Define Workflow
    # --------------------------------------------------------

    graph.add_edge(START, "data_upload")

    graph.add_edge("data_upload", "data_analysis")

    graph.add_edge("data_analysis", "problem_detection")

    graph.add_edge("problem_detection", "data_preparation")

    graph.add_edge("data_preparation", "model_selection")

    graph.add_edge("model_selection", "model_experimentation")

    graph.add_edge("model_experimentation", "training")

    graph.add_edge("training", "evaluation")

    graph.add_edge("evaluation", "reporting")

    graph.add_edge("reporting", END)

    # --------------------------------------------------------
    # Compile Graph
    # --------------------------------------------------------

    return graph.compile()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = create_graph()

    result = app.invoke({
        "file_path": "iris.csv",
        "target_column": "target"
    })

    print("\nMLPilot execution completed successfully.")