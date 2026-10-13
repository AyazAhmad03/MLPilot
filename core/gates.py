from core.config import settings


def data_quality_gate(state: dict) -> str:
    """Stands after data_analysis.
    Decides: is the dataset clean enough to continue, or should
    the cleaning agent run first?"""

    # Loop guard: if we already cleaned once, don't clean again.
    # Without this check -> analysis -> cleaning -> analysis -> cleaning ... forever.
    if state.get("cleaned"):
        return "ready"

    df = state["df"]
    total_cells = df.shape[0] * df.shape[1]
    total_missing = sum(state["missing_values"].values())

    # if the dataset has no cells at all, treat it as ready
    if total_cells == 0:
        return "ready"

    missing_ratio = total_missing / total_cells

    if missing_ratio > settings.missing_ratio_limit:
        return "needs_cleaning"
    return "ready"


def performance_gate(state: dict) -> str:
    """Stands after evaluation.
    Decides: is the model good enough, or should the tuning
    agent try to improve it first?"""

    # Loop guard: we only allow ONE tuning attempt.
    # Without this -> tuning -> training -> evaluation -> tuning ... forever.
    if not settings.enable_tuning or state.get("tuning_attempted"):
        return "report"

    metrics = state["metrics"]

    # classification report has "f1_score", regression report has "r2_score"
    threshold = (
        settings.classification_threshold
        if state["problem_type"] == "classification"
        else settings.regression_threshold
    )
    score = metrics.get("f1_score", metrics.get("r2_score", 0))

    if score < threshold:
        return "tune"
    return "report"