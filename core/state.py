from typing import TypedDict, Any


class GraphState(TypedDict, total=False):

    # Input
    file_path: str
    target_column: str

    # Dataset
    df: Any
    rows: int
    columns: int

    # Analysis
    missing_values: dict
    numerical_columns: list
    categorical_columns: list
    problem_type: str

    # ML data
    X: Any
    y: Any
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    preprocessor: Any

    # Models
    candidate_models: dict
    cv_results: dict
    best_model_name: str
    best_model: Any

    # Final model
    trained_pipeline: Any

    # Results
    metrics: dict
    report: str