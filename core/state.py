
from typing import TypedDict, Any


class GraphState(TypedDict, total=False):

    # ============================================================
    # INPUT
    # ============================================================
    file_path: str
    target_column: str

    # ============================================================
    # DATASET
    # ============================================================
    df: Any
    rows: int
    columns: int

    # ============================================================
    # DATA ANALYSIS
    # ============================================================
    missing_values: dict
    numerical_columns: list
    categorical_columns: list
    analysis_digest: str

    # Dataset diagnostics
    diagnostics: dict

    # Deterministic cleaning and preparation plans
    cleaning_plan: dict
    prep_plan: dict

    # Cleaning information
    dropped_columns: list
    cleaned: bool

    # ============================================================
    # PROBLEM DETECTION
    # ============================================================
    problem_type: str
    problem_spec: dict

    # ============================================================
    # DATA PREPARATION
    # ============================================================
    X: Any
    y: Any

    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any

    preprocessor: Any

    # Preparation metadata
    prep_metadata: dict

    # ============================================================
    # MODEL SELECTION
    # ============================================================
    candidate_models: dict

    # ============================================================
    # MODEL EXPERIMENTATION
    # ============================================================
    cv_results: dict
    best_model_name: str
    best_model: Any

    # ============================================================
    # MODEL TRAINING
    # ============================================================
    trained_pipeline: Any

    # ============================================================
    # HYPERPARAMETER TUNING
    # ============================================================
    tuning_attempted: bool
    tuning_history: list

    # ============================================================
    # MODEL EVALUATION
    # ============================================================
    metrics: dict

    # ============================================================
    # MLFLOW TRACKING
    # ============================================================
    mlflow_run_id: str
    mlflow_tracking_uri: str
    mlflow_status: str
    mlflow_error: str

    # ============================================================
    # AI INSIGHTS
    # ============================================================
    insight: str

    # ============================================================
    # FINAL REPORT
    # ============================================================
    report: str
