from core.state import GraphState


# ============================================================
# 9. AUTOMATED REPORT AGENT
# ============================================================

def report_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("REPORT AGENT")
    print("=" * 60)

    problem_type = state["problem_type"]

    best_model = state["best_model_name"]

    rows = state["rows"]

    columns = state["columns"]

    metrics = state["metrics"]

    cv_results = state["cv_results"]

    report = f"""

============================================================
                    MLPilot REPORT
============================================================

DATASET
------------------------------------------------------------
Rows              : {rows}
Columns           : {columns}

PROBLEM
------------------------------------------------------------
Problem Type      : {problem_type}

MODEL SELECTION
------------------------------------------------------------
Best Model        : {best_model}

CROSS VALIDATION
------------------------------------------------------------
"""

    for model_name, result in cv_results.items():

        report += (
            f"{model_name}: "
            f"{result['mean_score']:.4f}\n"
        )

    report += """

FINAL TEST PERFORMANCE
------------------------------------------------------------
"""

    for metric, value in metrics.items():

        report += (
            f"{metric}: "
            f"{value:.4f}\n"
        )

    report += """
============================================================
                  END OF REPORT
============================================================
"""

    print(report)

    return {
        "report": report
    }