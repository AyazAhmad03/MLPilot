import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from core.state import GraphState


def evaluation_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("MODEL EVALUATION AGENT")
    print("=" * 60)

    X_test = state["X_test"]
    y_test = state["y_test"]

    pipeline = state["trained_pipeline"]
    problem_type = state["problem_type"]

    # Make predictions
    predictions = pipeline.predict(X_test)

    metrics = {}

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if problem_type == "classification":

        metrics["accuracy"] = accuracy_score(
            y_test,
            predictions
        )

        metrics["precision"] = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        metrics["recall"] = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        metrics["f1_score"] = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    else:

        metrics["mae"] = mean_absolute_error(
            y_test,
            predictions
        )

        metrics["rmse"] = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        metrics["r2_score"] = r2_score(
            y_test,
            predictions
        )

    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

    print("\nFinal Test Results:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # --------------------------------------------------------
    # PATCH C: MLflow tracking (optional — never crashes the pipeline)
    # --------------------------------------------------------

    try:
        import mlflow

        with mlflow.start_run(run_name=state.get("best_model_name", "mlpilot_run")):
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param("rows", state.get("rows"))
            mlflow.log_param("candidates_tested", len(state.get("cv_results", {})))
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.sklearn.log_model(
                state.get("trained_pipeline"),
                name="model",
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
        print("Run tracked in MLflow (mlruns/).")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

    return {
        "metrics": metrics
    }