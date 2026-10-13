import logging

import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

from core.config import settings

logger = logging.getLogger("mlpilot")

# stop optuna from printing one line per trial (console spam)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# models that actually have hyperparameters worth tuning
TUNABLE_MODELS = {"Random Forest", "Logistic Regression", "Decision Tree", "SVC", "SVR", "Ridge Regression"}


def build_model(name: str, trial, problem: str):
    """Maps (model name + Optuna trial) -> a model with settings suggested
    by Optuna for THIS trial. Each trial gets different values."""
    if name == "Random Forest":
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 25),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        )
        cls = RandomForestClassifier if problem == "classification" else RandomForestRegressor
        return cls(random_state=settings.random_state, **params)

    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000,
            C=trial.suggest_float("C", 1e-3, 100, log=True),
        )

    if name == "Decision Tree":
        params = dict(
            max_depth=trial.suggest_int("max_depth", 2, 25),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        )
        cls = DecisionTreeClassifier if problem == "classification" else DecisionTreeRegressor
        return cls(random_state=settings.random_state, **params)

    if name in ("SVC", "SVR"):
        params = dict(
            C=trial.suggest_float("C", 1e-2, 100, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
        )
        if problem == "classification":
            return SVC(probability=True, **params)
        return SVR(**params)

    if name == "Ridge Regression":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))

    return None


# ================================================================
# >>> FROM HERE DOWN IS THE ONLY CHANGED PART <<<
# (old tuning_agent: DELETE it, paste this new one)
# ================================================================

def tuning_agent(state: dict) -> dict:
    """The pit crew. Tunes the winning model with Optuna.
    REGRESSION GUARD: the tuned model is adopted ONLY if it beats
    the original CV score — otherwise the original model is kept."""

    problem = state["problem_type"]
    name = state["best_model_name"]
    history = state.get("tuning_history", [])

    # ---- Skip models with nothing to tune (e.g. Linear Regression) ----
    if name not in TUNABLE_MODELS:
        logger.info(f"{name} has no hyperparameters worth tuning — skipping Optuna")
        return {
            "tuning_attempted": True,   # flag set anyway -> loop still ends
            "tuning_history": history,
        }

    scoring = "f1_weighted" if problem == "classification" else "r2"
    logger.info(f"Tuning '{name}' with Optuna ({settings.tuning_trials} trials)")

    # ---- The objective: Optuna maximizes whatever this returns ----
    def objective(trial):
        model = build_model(name, trial, problem)
        if model is None:
            raise ValueError(f"No search space defined for model: {name}")

        # preprocessor INSIDE the pipeline -> no leakage during tuning either
        pipe = Pipeline([
            ("preprocessor", state["preprocessor"]),
            ("model", model),
        ])
        return cross_val_score(
            pipe,
            state["X_train"],
            state["y_train"],
            cv=settings.cv_folds,
            scoring=scoring,
            n_jobs=-1,
        ).mean()

    study = optuna.create_study(direction="maximize")  # higher score = better
    study.optimize(objective, n_trials=settings.tuning_trials)

    old_score = state.get("cv_results", {}).get(name, {}).get("mean_score")
    new_score = study.best_value

    # ---- REGRESSION GUARD: keep original if tuning didn't help ----
    if old_score is not None and new_score <= old_score:
        logger.info(
            f"Tuning did NOT improve CV score "
            f"({old_score:.4f} -> {new_score:.4f}). Keeping the original model."
        )
        return {
            "best_model": state.get("best_model"),   # keep the ORIGINAL model
            "best_model_name": name,                 # no "(tuned)" tag
            "tuning_attempted": True,                # loop still ends
            "tuning_history": history + [
                {"model": name, "params": study.best_params,
                 "cv_score": round(new_score, 4), "accepted": False}
            ],
        }

    # ---- Tuning won: adopt the improved model ----
    tuned_model = build_model(name, study.best_trial, problem)
    if old_score is not None:
        logger.info(f"CV score improved: {old_score:.4f} -> {new_score:.4f}")
    else:
        logger.info(f"Best CV score: {new_score:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return {
        "best_model": tuned_model,                    # training agent will fit this
        "best_model_name": f"{name} (tuned)",         # so the report shows it
        "tuning_attempted": True,                     # LOOP GUARD for gate 2
        "tuning_history": history + [
            {"model": name, "params": study.best_params,
             "cv_score": round(new_score, 4), "accepted": True}
        ],
    }