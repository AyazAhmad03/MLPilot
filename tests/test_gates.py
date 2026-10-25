"""
Unit tests for MLPilot's two decision gates.

The gates are pure functions: state in -> decision out. No training,
no MLflow needed. We build tiny fake states and check every routing
decision — especially the two loop terminators that prevent loops.
"""

import pandas as pd

from core.config import settings
from core.gates import data_quality_gate, performance_gate


# ------------------------------------------------------------------
# Helpers: build the exact inputs the real gates read.
# ------------------------------------------------------------------
def make_df(missing_ratio=0.0):
    """A tiny 10-row dataset; missing_ratio controls what fraction is NaN."""
    df = pd.DataFrame({
        "age":    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        "salary": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    })
    k = int(10 * missing_ratio)              # how many rows to blank out
    if k:
        df.loc[df.index[:k], ["age", "salary"]] = None
    return df


def make_state(**overrides):
    """Minimal fake state containing every key the gates touch."""
    state = {
        "cleaned": False,                     # loop guard flag
        "df": make_df(0.0),                   # 10 rows x 2 cols = 20 cells
        "missing_values": {},                 # column -> missing COUNT (not ratio)
        "target_missing": 0,
        "problem_type": "classification",
        "metrics": {"f1_score": 0.96},
        "tuning_attempted": False,            # loop guard flag
    }
    state.update(overrides)
    return state


# ---------------- Gate 1: data_quality_gate ----------------

def test_dirty_data_routes_to_cleaning():
    """14 of 20 cells missing (70%) + not cleaned yet -> cleaning must run."""
    state = make_state(df=make_df(0.7),
                       missing_values={"age": 7, "salary": 7})   # 7+7 = 14
    assert data_quality_gate(state) == "needs_cleaning"


def test_cleaned_flag_terminates_cleaning_loop():
    """THE loop terminator: cleaned=True -> ready, even if data is filthy."""
    state = make_state(cleaned=True,
                       df=make_df(0.7),
                       missing_values={"age": 7, "salary": 7})
    assert data_quality_gate(state) == "ready"


def test_clean_data_skips_cleaning():
    """Zero missing cells -> straight to preparation."""
    state = make_state(df=make_df(0.0), missing_values={})
    assert data_quality_gate(state) == "ready"


# ---------------- Gate 2: performance_gate ----------------

def test_weak_model_triggers_tuning():
    """F1 0.60 < 0.75 threshold + tuning not tried yet -> tune."""
    state = make_state(metrics={"f1_score": 0.60})
    assert performance_gate(state) == "tune"


def test_tuning_attempted_terminates_tuning_loop():
    """THE loop terminator: tuning already attempted -> report, even if weak."""
    state = make_state(metrics={"f1_score": 0.60}, tuning_attempted=True)
    assert performance_gate(state) == "report"


def test_strong_model_skips_tuning():
    """F1 0.96 is above the threshold -> straight to report."""
    state = make_state(metrics={"f1_score": 0.96})
    assert performance_gate(state) == "report"


def test_disabled_tuning_goes_to_report(monkeypatch):
    """Tuning off in config.yaml -> never tune, even with a weak model."""
    monkeypatch.setattr(settings, "enable_tuning", False)
    state = make_state(metrics={"f1_score": 0.60})
    assert performance_gate(state) == "report"