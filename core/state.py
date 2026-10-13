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
    dropped_columns: list
    cleaned: bool              # NEW — prevents infinite cleaning loop

    # Problem
    problem_type: str

    # ML data
    X: Any
    y: Any
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    preprocessor: Any          # FIX — was missing before

    # Models
    candidate_models: dict
    cv_results: dict
    best_model_name: str
    best_model: Any

    # Tuning loop
    tuning_attempted: bool     # NEW
    tuning_history: list       # NEW

    # Final
    trained_pipeline: Any
    metrics: dict
    report: str