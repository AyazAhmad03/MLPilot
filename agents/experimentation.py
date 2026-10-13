import logging

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from core.config import settings
from core.state import GraphState

logger = logging.getLogger("mlpilot")


def model_experimentation_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("MODEL EXPERIMENTATION AGENT")
    print("=" * 60)

    X_train = state["X_train"]
    y_train = state["y_train"]
    preprocessor = state["preprocessor"]
    candidate_models = state["candidate_models"]
    problem_type = state["problem_type"]

    cv_results = {}
    best_model_name = None
    best_model = None
    best_score = -np.inf

    if problem_type == "classification":
        scoring = "f1_weighted"
    else:
        scoring = "r2"

    for model_name, model in candidate_models.items():

        print(f"\nTesting: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=settings.cv_folds,      # UPGRADE: was hardcoded cv=5
            scoring=scoring
        )

        mean_score = scores.mean()

        cv_results[model_name] = {
            "scores": scores.tolist(),
            "mean_score": mean_score
        }

        print(f"CV Score: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = model_name
            best_model = model

    print(f"\nBest model: {best_model_name}")
    print(f"Best CV score: {best_score:.4f}")

    return {
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "best_model": best_model
    }