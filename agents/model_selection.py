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


# ============================================================
# 6. MODEL EXPERIMENTATION AGENT
# ============================================================

def model_experimentation_agent(
    state: GraphState
):

    print("\n" + "=" * 60)
    print("MODEL EXPERIMENTATION AGENT")
    print("=" * 60)

    X_train = state["X_train"]

    y_train = state["y_train"]

    preprocessor = state["preprocessor"]

    candidate_models = (
        state["candidate_models"]
    )

    problem_type = state["problem_type"]

    cv_results = {}

    best_model_name = None

    best_model = None

    best_score = -np.inf

    # Select scoring metric

    if problem_type == "classification":

        scoring = "f1_weighted"

    else:

        scoring = "r2"

    # Test every candidate model

    for model_name, model in candidate_models.items():

        print(
            f"\nTesting: {model_name}"
        )

        pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        scores = cross_val_score(

            pipeline,

            X_train,

            y_train,

            cv=5,

            scoring=scoring
        )

        mean_score = scores.mean()

        cv_results[model_name] = {

            "scores":
                scores.tolist(),

            "mean_score":
                mean_score
        }

        print(
            f"CV Score: "
            f"{mean_score:.4f}"
        )

        # Select best model

        if mean_score > best_score:

            best_score = mean_score

            best_model_name = model_name

            best_model = model

    print(
        f"\nBest model: "
        f"{best_model_name}"
    )

    print(
        f"Best CV score: "
        f"{best_score:.4f}"
    )

    return {
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "best_model": best_model
    }