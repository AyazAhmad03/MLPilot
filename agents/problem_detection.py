import logging

from core.state import GraphState

logger = logging.getLogger("mlpilot")


def problem_detection_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("PROBLEM DETECTION AGENT")
    print("=" * 60)

    df = state["df"]
    target_column = state["target_column"]

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    target = df[target_column]
    unique_values = target.nunique()

    if (
        target.dtype == "object"
        or str(target.dtype) == "category"
        or target.dtype == "bool"
        or unique_values <= 10
    ):
        problem_type = "classification"
    else:
        problem_type = "regression"

    print(f"Target column: {target_column}")
    print(f"Unique target values: {unique_values}")
    print(f"Detected problem type: {problem_type}")

    return {
        "problem_type": problem_type
    }