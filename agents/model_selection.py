import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.svm import (
    SVC,
    SVR
)

from core.state import GraphState


# ============================================================
# 5. INTELLIGENT MODEL SELECTION AGENT
# ============================================================

def model_selection_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("MODEL SELECTION AGENT")
    print("=" * 60)

    problem_type = state["problem_type"]

    rows = state["rows"]

    candidate_models = {}

    # Classification

    if problem_type == "classification":

        candidate_models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "SVC":
                SVC()
        }

        if rows > 10000:

            candidate_models.pop(
                "SVC"
            )

    # Regression

    else:

        candidate_models = {

            "Linear Regression":
                LinearRegression(),

            "Ridge Regression":
                Ridge(),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                ),

            "Decision Tree":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "SVR":
                SVR()
        }

        if rows > 10000:

            candidate_models.pop(
                "SVR"
            )

    print("\nCandidate models:")

    for model_name in candidate_models:

        print(
            f"  - {model_name}"
        )

    return {
        "candidate_models": candidate_models
    }
