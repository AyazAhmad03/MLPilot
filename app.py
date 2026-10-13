import os
import tempfile

import pandas as pd
import streamlit as st

from graph import build_graph

st.set_page_config(page_title="MLPilot", page_icon="🛩️", layout="wide")
st.title("🛩️ MLPilot — Agentic AutoML Pipeline")
st.caption("LangGraph pipeline with two feedback loops: a data-quality gate and a performance gate")


@st.cache_resource
def load_app():
    return build_graph()


app = load_app()

# ---------------- Sidebar: inputs ----------------
st.sidebar.header("Input")
source = st.sidebar.radio(
    "Dataset",
    ["Demo: iris (clean)", "Demo: iris_messy (broken)", "Demo: customers (IDs + dates)", "Upload my own CSV"],
)

demo_files = {
    "Demo: iris (clean)": ("iris.csv", "target"),
    "Demo: iris_messy (broken)": ("iris_messy.csv", "target"),
    "Demo: customers (IDs + dates)": ("customers.csv", "churn"),
}

if source == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    target = st.sidebar.text_input("Target column name")
    file_path = None
    if uploaded is not None and target:
        file_path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())
else:
    file_path, target = demo_files[source]
    st.sidebar.info(f"Running `{file_path}` with target `{target}`")

run = st.sidebar.button("🚀 Run MLPilot", type="primary", disabled=not (file_path and target))

# ---------------- Run the graph ----------------
if run:
    inputs = {"file_path": file_path, "target_column": target}
    final_state = {}
    visited = []

    with st.status("Running MLPilot agents...", expanded=True) as status:
        for chunk in app.stream(inputs, config={"recursion_limit": 50}):
            for node_name, update in chunk.items():
                visited.append(node_name)
                st.write(f"✅ `{node_name}`")
                final_state.update(update)
        status.update(label="Pipeline complete!", state="complete")

    # ---- The agentic part: did the loops fire? ----
    col1, col2 = st.columns(2)
    with col1:
        if "data_cleaning" in visited:
            st.warning("🔄 **Data-quality loop FIRED** — data was cleaned and re-analyzed")
        else:
            st.success("Data-quality gate: skipped (data was clean enough)")
    with col2:
        if "tuning" in visited:
            st.warning("🔄 **Performance loop FIRED** — model was tuned with Optuna and re-evaluated")
        else:
            st.success("Performance gate: skipped (score above threshold)")

    # ---- Metrics ----
    st.subheader(f"🏆 Best model: `{final_state.get('best_model_name', '?')}`")
    metrics = final_state.get("metrics", {})
    if metrics:
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}")

    # ---- CV comparison chart ----
    cv = final_state.get("cv_results", {})
    if cv:
        st.subheader("Cross-validation comparison")
        st.bar_chart(
            pd.DataFrame({m: r["mean_score"] for m, r in cv.items()}, index=["CV score"]).T
        )

    # ---- Tuning history (if the loop ran) ----
    history = final_state.get("tuning_history", [])
    if history:
        st.subheader("Optuna tuning results")
        st.json(history)

    # ---- Report ----
    st.subheader("Full report")
    st.code(final_state.get("report", "No report generated"), language="text")

    # ---- Model download ----
    if os.path.exists("artifacts/best_model.joblib"):
        with open("artifacts/best_model.joblib", "rb") as f:
            st.download_button("⬇️ Download trained model (.joblib)", f, "best_model.joblib")