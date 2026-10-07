from sklearn.pipeline import Pipeline

from core.state import GraphState


# ============================================================
# 7. TRAINING AGENT
# ============================================================

def training_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("MODEL TRAINING AGENT")
    print("=" * 60)

    X_train = state["X_train"]

    y_train = state["y_train"]

    preprocessor = state["preprocessor"]

    best_model = state["best_model"]

    trained_pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                best_model
            )
        ]
    )

    print(
        f"Training model: "
        f"{state['best_model_name']}"
    )

    trained_pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Model trained successfully."
    )

    return {
        "trained_pipeline":
            trained_pipeline
    }