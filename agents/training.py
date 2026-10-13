import os

import joblib
from sklearn.pipeline import Pipeline

from core.state import GraphState


def training_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("MODEL TRAINING AGENT")
    print("=" * 60)

    X_train = state["X_train"]
    y_train = state["y_train"]
    preprocessor = state["preprocessor"]
    best_model = state["best_model"]

    trained_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", best_model),
    ])

    print(f"Training model: {state['best_model_name']}")
    trained_pipeline.fit(X_train, y_train)
    print("Model trained successfully.")

    # ---- persist the artifact so it can be served later (API/UI) ----
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(trained_pipeline, "artifacts/best_model.joblib")
    print("Model saved to artifacts/best_model.joblib")

    return {"trained_pipeline": trained_pipeline}