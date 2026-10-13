"""MLPilot - Streamlit interface for the agentic AutoML graph.

Run from the project root:  streamlit run app.py
Keep .streamlit/config.toml in the folder you run the command from.
"""
import html
import os
import tempfile
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from graph import build_graph

st.set_page_config(page_title="MLPilot", page_icon="🧭", layout="wide")

AMBER, VIOLET, LINE, MUTED = "#F3B562", "#6C5FB5", "#322A58", "#9A92BD"

# --------------------------------------------------------------------------
# Styling and animation
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
:root{--bg:#130F26;--panel:#1B1636;--line:#322A58;--text:#ECE8F7;--muted:#9A92BD;--amber:#F3B562;--mint:#7FD6B2;--rose:#E88BA0;--violet:#6C5FB5}
.stApp{font-family:'Instrument Sans',system-ui,sans-serif;background:radial-gradient(900px 420px at 88% -10%,#261D4D 0%,transparent 65%),var(--bg)}
h1,h2,h3,.ht,.sec,.cv,.gt{font-family:'Bricolage Grotesque',sans-serif!important;letter-spacing:-.01em}
#MainMenu,footer,.stDeployButton,[data-testid="stToolbar"]{display:none!important}
header[data-testid="stHeader"]{background:transparent;height:0}
.block-container{padding-top:2rem;max-width:1220px}
section[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line)}

/* Header */
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:18px;padding:20px 26px;
  background:linear-gradient(120deg,#231B4A,#181331 55%,#201840);background-size:200% 200%;animation:drift 18s ease-in-out infinite}
.hero::after{content:"";position:absolute;right:-60px;top:-90px;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(243,181,98,.16),transparent 65%);pointer-events:none}
@keyframes drift{50%{background-position:100% 50%}}
.hrow{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.hl{display:flex;align-items:center;gap:14px}
.mark{width:44px;height:44px;border-radius:13px;display:grid;place-items:center;font:700 17px 'Bricolage Grotesque',sans-serif;color:#2A1A05;background:linear-gradient(135deg,#F7C77E,#E08F3E);box-shadow:0 6px 22px rgba(243,181,98,.28)}
.ht{font-size:25px;font-weight:700;line-height:1.1}
.hs{color:var(--muted);font-size:13.5px;margin-top:3px}
.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:5px 14px;font-size:13px;color:var(--muted);background:rgba(19,15,38,.6)}
.pill i{width:8px;height:8px;border-radius:50%;background:var(--muted);display:inline-block}
.pill.fly,.pill.done,.pill.err{color:var(--text)}
.pill.fly i{background:var(--amber);animation:blink 1s ease-in-out infinite}
.pill.done i{background:var(--mint)} .pill.err i{background:var(--rose)}
@keyframes blink{50%{opacity:.25}}

/* Pipeline map */
.pl{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 22px;margin-top:16px;overflow-x:auto}
.plh{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;min-width:640px}
.plt{color:var(--muted);font-size:11px;letter-spacing:2.2px;font-weight:600}
.plc{color:var(--amber);font-size:12px;font-weight:600}
.bar{height:4px;background:var(--bg);border-radius:99px;overflow:hidden;margin-bottom:12px;min-width:640px}
.bar div{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--mint),var(--amber));transition:width .7s cubic-bezier(.2,.8,.2,1)}
.row{display:flex;align-items:center;margin:10px 0;min-width:640px}
.stg{color:var(--muted);font-size:10.5px;letter-spacing:2px;width:84px;flex-shrink:0;font-weight:600}
.n{display:flex;align-items:center;gap:7px;padding:7px 13px;border-radius:10px;font-size:12px;font-weight:600;white-space:nowrap;border:1.5px solid var(--line);background:var(--bg);color:var(--muted);transition:all .4s}
.n.done{background:rgba(127,214,178,.1);border-color:rgba(127,214,178,.6);color:var(--text)}
.n.on{background:rgba(243,181,98,.12);border-color:var(--amber);color:#fff;animation:ring 1.4s ease-out infinite}
.tk{color:var(--mint);font-size:11px}
.lp{font-size:10px;color:var(--amber);border:1px solid var(--amber);border-radius:99px;padding:0 7px;font-weight:700}
.lk{flex:1;min-width:14px;height:2px;margin:0 7px;background:var(--line);border-radius:2px;transition:background .5s}
.lk.done{background:var(--mint)}
.lk.flow{background:linear-gradient(90deg,var(--mint),var(--amber),var(--mint));background-size:200% 100%;animation:flow 1s linear infinite}
@keyframes flow{to{background-position:-200% 0}}
@keyframes ring{0%{box-shadow:0 0 0 0 rgba(243,181,98,.5)}100%{box-shadow:0 0 0 10px rgba(243,181,98,0)}}
.loops{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
.lc{font-size:12px;color:var(--muted);border:1px dashed var(--line);border-radius:99px;padding:3px 12px;transition:all .4s}
.lc.on{color:var(--amber);border:1px solid var(--amber);background:rgba(243,181,98,.08)}

/* Cards */
.sec{font-size:19px;font-weight:700;margin:30px 0 4px}
.hint{color:var(--muted);font-size:14px;margin:0 0 12px}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.card span{display:block;color:var(--muted);font-size:12.5px;letter-spacing:.4px}
.card .cv{display:block;font-size:27px;font-weight:700;margin-top:4px;color:var(--text);letter-spacing:-.01em}
.card.hi .cv{color:var(--amber)}
.card.fired{border-color:var(--amber);background:linear-gradient(135deg,rgba(243,181,98,.1),var(--panel))}
.card.pass{border-color:rgba(127,214,178,.5);background:linear-gradient(135deg,rgba(127,214,178,.08),var(--panel))}
.gt{font-size:15px;font-weight:700}
.card.fired .gt{color:var(--amber)} .card.pass .gt{color:var(--mint)}
.card p{margin:4px 0 0;color:var(--muted);font-size:13px}
.rise{animation:rise .55s cubic-bezier(.2,.8,.2,1) both;animation-delay:var(--d,0s)}
@keyframes rise{from{opacity:0;transform:translateY(10px)}}

/* Widgets */
button[kind="primary"],[data-testid="stBaseButton-primary"]{background:var(--amber)!important;border-color:var(--amber)!important;color:#2A1A05!important;font-weight:600;transition:transform .15s,filter .15s}
button[kind="primary"]:hover,[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.08);transform:translateY(-1px)}
[data-baseweb="select"]>div:focus-within,[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--amber)!important}
[data-testid="stExpander"],[data-testid="stStatusWidget"]{border:1px solid var(--line);border-radius:12px;background:var(--panel)}
.stTabs [aria-selected="true"]{color:var(--amber)!important}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_app():
    return build_graph()


app = load_app()

# --------------------------------------------------------------------------
# Pipeline definition
# --------------------------------------------------------------------------
TOP = [("data_upload", "UPLOAD"), ("data_analysis", "ANALYSIS"), ("data_cleaning", "CLEANING"),
       ("problem_detection", "DETECTION"), ("data_preparation", "PREPARATION")]
BOTTOM = [("model_selection", "SELECTION"), ("model_experimentation", "BENCHMARK"), ("training", "TRAINING"),
          ("evaluation", "EVALUATION"), ("tuning", "TUNING"), ("report", "REPORT")]
ALL = [n for n, _ in TOP + BOTTOM]
LOOP_NODES = {"data_cleaning", "tuning"}


def header_html(kind: str, label: str) -> str:
    return (
        '<div class="hero"><div class="hrow"><div class="hl"><div class="mark">ML</div>'
        '<div><div class="ht">MLPilot</div>'
        '<div class="hs">Agentic AutoML pipeline · two feedback loops · Optuna tuning · MLflow tracking</div></div></div>'
        f'<span class="pill {kind}"><i></i>{label}</span></div></div>'
    )


def map_html(counts: Counter, latest: str | None) -> str:
    visited = set(counts)

    def node(name, label):
        state = "on" if name == latest else ("done" if name in visited else "")
        tick = '<span class="tk">✓</span>' if name in visited and name != latest else ""
        loop = f'<span class="lp">↺ ×{counts[name]}</span>' if counts[name] > 1 else ""
        return f'<div class="n {state}">{label}{tick}{loop}</div>'

    def row(stage, nodes):
        parts = [f'<div class="stg">{stage}</div>']
        for i, (name, label) in enumerate(nodes):
            parts.append(node(name, label))
            if i < len(nodes) - 1:
                nxt = nodes[i + 1][0]
                cls = "flow" if name == latest else ("done" if nxt in visited else "")
                parts.append(f'<div class="lk {cls}"></div>')
        return f'<div class="row">{"".join(parts)}</div>'

    pct = int(len(visited) / len(ALL) * 100)
    dq = "on" if "data_cleaning" in visited else ""
    pf = "on" if "tuning" in visited else ""
    return (
        '<div class="pl"><div class="plh"><div class="plt">PIPELINE EXECUTION</div>'
        f'<div class="plc">{len(visited)} / {len(ALL)} agents</div></div>'
        f'<div class="bar"><div style="width:{pct}%"></div></div>'
        f'{row("DATA", TOP)}{row("MODEL", BOTTOM)}'
        f'<div class="loops"><span class="lc {dq}">↺ Data-quality loop · Cleaning returns to Analysis</span>'
        f'<span class="lc {pf}">↺ Performance loop · Tuning returns to Training</span></div></div>'
    )


def cards_html(items, extra: str = "", delay: float = 0.0) -> str:
    out = []
    for i, (label, value) in enumerate(items):
        out.append(f'<div class="card rise {extra}" style="--d:{delay + i * 0.08:.2f}s"><span>{html.escape(str(label))}</span>'
                   f'<span class="cv">{html.escape(str(value))}</span></div>')
    return f'<div class="g">{"".join(out)}</div>'


def gate_html(fired: bool, name: str, detail: str, delay: float) -> str:
    cls = "fired" if fired else "pass"
    title = f"↺ {name} · fired" if fired else f"✓ {name} · skipped"
    return (f'<div class="card rise {cls}" style="--d:{delay:.2f}s"><div class="gt">{title}</div>'
            f'<p>{html.escape(detail)}</p></div>')


def section(title: str, hint: str = "") -> None:
    st.markdown(f'<div class="sec">{title}</div><div class="hint">{hint}</div>', unsafe_allow_html=True)


def describe(node: str, u: dict) -> str:
    try:
        if node == "data_upload":
            return f"{u['rows']} rows × {u['columns']} cols"
        if node == "data_analysis":
            return f"{len(u.get('missing_values', {}))} cols with missing values"
        if node == "data_cleaning":
            return f"dropped {len(u.get('dropped_columns', []))} cols → {u['rows']} rows"
        if node == "problem_detection":
            return u["problem_type"]
        if node == "data_preparation":
            return "split + preprocessor ready"
        if node == "model_selection":
            return f"{len(u['candidate_models'])} candidates"
        if node == "model_experimentation":
            return f"best: {u['best_model_name']}"
        if node == "training":
            return f"fitted {u.get('best_model_name', '')}"
        if node == "tuning":
            h = u.get("tuning_history", [])
            if h:
                return f"CV {h[-1]['cv_score']:.4f} · " + ("accepted" if h[-1].get("accepted") else "rejected")
        if node == "evaluation":
            return ", ".join(f"{k} {v:.3f}" for k, v in u.get("metrics", {}).items())
        if node == "report":
            return "generated"
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
DEMOS = {
    "Demo: iris (clean)": ("iris.csv", "target"),
    "Demo: iris_messy (broken)": ("iris_messy.csv", "target"),
    "Demo: customers (IDs + dates)": ("customers.csv", "churn"),
}

with st.sidebar:
    st.markdown("### Setup")
    source = st.selectbox("Dataset", list(DEMOS) + ["Upload my own CSV"])
    file_path, target = None, None
    if source == "Upload my own CSV":
        uploaded = st.file_uploader("CSV file", type="csv")
        if uploaded is not None:
            try:
                cols = list(pd.read_csv(uploaded, nrows=0).columns)
                uploaded.seek(0)
                target = st.selectbox("Target column", cols, index=len(cols) - 1)
                file_path = os.path.join(tempfile.gettempdir(), os.path.basename(uploaded.name))
                with open(file_path, "wb") as f:
                    f.write(uploaded.getbuffer())
            except Exception as exc:
                st.error(f"Could not read that file: {exc}")
    else:
        file_path, target = DEMOS[source]
        st.caption(f"`{file_path}` · target `{target}`")
    run = st.button("Run MLPilot", type="primary", use_container_width=True, disabled=not (file_path and target))

# --------------------------------------------------------------------------
# Header + live pipeline map
# --------------------------------------------------------------------------
if "run" not in st.session_state:
    st.session_state.run = None

header_slot = st.empty()
map_slot = st.empty()
saved = st.session_state.run

if saved and not run:
    header_slot.markdown(header_html("done", "Complete"), unsafe_allow_html=True)
    map_slot.markdown(map_html(saved["counts"], None), unsafe_allow_html=True)
else:
    header_slot.markdown(header_html("", "Standing by"), unsafe_allow_html=True)
    map_slot.markdown(map_html(Counter(), None), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if run:
    st.session_state.run = None
    updates, counts, latest = {}, Counter(), None
    header_slot.markdown(header_html("fly", "Running"), unsafe_allow_html=True)
    try:
        with st.status("Running agents…", expanded=True) as status:
            for chunk in app.stream({"file_path": file_path, "target_column": target},
                                    config={"recursion_limit": 50}):
                for node, update in chunk.items():
                    counts[node] += 1
                    updates[node] = update
                    latest = node
                    map_slot.markdown(map_html(counts, latest), unsafe_allow_html=True)
                    detail = describe(node, update)
                    st.write(f"**{node}**" + (f" · {detail}" if detail else ""))
            status.update(label="Pipeline complete", state="complete", expanded=False)
        final = {}
        for u in updates.values():
            final.update(u)
        st.session_state.run = {"final": final, "updates": updates, "counts": counts}
        header_slot.markdown(header_html("done", "Complete"), unsafe_allow_html=True)
        map_slot.markdown(map_html(counts, None), unsafe_allow_html=True)
        st.toast("Pipeline complete", icon="✅")
    except Exception as exc:
        header_slot.markdown(header_html("err", "Stopped"), unsafe_allow_html=True)
        st.error(f"The pipeline stopped: {exc}")

# --------------------------------------------------------------------------
# Results (kept in session state so downloads don't wipe them)
# --------------------------------------------------------------------------
saved = st.session_state.run
if not saved:
    st.info("Choose a dataset in the sidebar and select **Run MLPilot**.")
    st.stop()

final, updates, counts = saved["final"], saved["updates"], saved["counts"]
cl = updates.get("data_cleaning")
tuned = "tuning" in counts

section("Feedback loops", "Whether each quality gate sent the run back for another pass.")
st.markdown(
    '<div class="g">'
    + gate_html(cl is not None, "Data-quality loop",
                "Data was cleaned and re-analysed." if cl else "Data was clean enough.", 0.0)
    + gate_html(tuned, "Performance loop",
                "Optuna tuned the model and it was re-evaluated." if tuned else "Score cleared the threshold.", 0.1)
    + "</div>", unsafe_allow_html=True)

metrics = final.get("metrics", {})
section("Best model", "Scores on the held-out test split.")
items = [("Model", final.get("best_model_name", "?"))] + [(k.replace("_", " ").title(), f"{v:.4f}") for k, v in metrics.items()]
st.markdown(cards_html(items, "hi", 0.15), unsafe_allow_html=True)

st.write("")
t_rep, t_data, t_model, t_opt = st.tabs(["Report", "Data inspector", "Model internals", "Optuna"])

with t_rep:
    st.code(final.get("report", ""), language="text")

with t_data:
    if cl is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows before", updates.get("data_upload", {}).get("rows", "—"))
        c2.metric("Rows after", cl.get("rows", "—"))
        c3.metric("Columns dropped", len(cl.get("dropped_columns", [])))
        if cl.get("dropped_columns"):
            st.caption("Dropped: " + ", ".join(cl["dropped_columns"]))
    else:
        st.info("The quality gate was skipped, so nothing needed cleaning.")
    st.dataframe(final.get("df", pd.DataFrame()).head(10), use_container_width=True)

with t_model:
    cv = final.get("cv_results", {})
    best = final.get("best_model_name")
    if cv:
        cv_df = pd.DataFrame([{"model": m, "score": r["mean_score"], "best": m == best} for m, r in cv.items()])
        st.markdown("**Cross-validation scores**")
        st.altair_chart(
            alt.Chart(cv_df).mark_bar(cornerRadiusEnd=4, size=22).encode(
                x=alt.X("score:Q", title=None, scale=alt.Scale(domain=[min(0.0, float(cv_df.score.min())), 1]),
                        axis=alt.Axis(labelColor=MUTED, gridColor=LINE, domain=False)),
                y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelColor="#ECE8F7", ticks=False, domain=False)),
                color=alt.condition(alt.datum.best, alt.value(AMBER), alt.value(VIOLET)),
                tooltip=["model", alt.Tooltip("score:Q", format=".4f")],
            ).properties(height=44 * len(cv_df) + 20, background="transparent").configure_view(strokeWidth=0),
            use_container_width=True)

    pipe = final.get("trained_pipeline")
    if pipe is not None and final.get("problem_type") == "classification":
        try:
            labels = sorted(pd.unique(final["y_test"]))
            cm = confusion_matrix(final["y_test"], pipe.predict(final["X_test"]), labels=labels)
            st.markdown("**Confusion matrix (test set)**")
            st.dataframe(pd.DataFrame(cm, index=[f"true {l}" for l in labels],
                                      columns=[f"pred {l}" for l in labels]), use_container_width=True)
        except Exception:
            pass
    try:
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        model = pipe.named_steps["model"]
        imp = getattr(model, "feature_importances_", None)
        if imp is None and hasattr(model, "coef_"):
            imp = np.abs(model.coef_).flatten()
        if imp is not None:
            imp_df = (pd.DataFrame({"feature": names, "importance": imp})
                      .sort_values("importance", ascending=False).head(12))
            st.markdown("**Top feature importance**")
            st.altair_chart(
                alt.Chart(imp_df).mark_bar(cornerRadiusEnd=4, size=18, color=AMBER).encode(
                    x=alt.X("importance:Q", title=None, axis=alt.Axis(labelColor=MUTED, gridColor=LINE, domain=False)),
                    y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelColor="#ECE8F7", ticks=False, domain=False)),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
                ).properties(height=30 * len(imp_df) + 20, background="transparent").configure_view(strokeWidth=0),
                use_container_width=True)
    except Exception:
        st.caption("Feature importance is not available for this model.")

with t_opt:
    hist = final.get("tuning_history", [])
    if hist:
        try:
            st.dataframe(pd.json_normalize(hist), use_container_width=True)
        except Exception:
            st.json(hist)
    else:
        st.info("The tuning loop did not run. The first pass satisfied the gate.")

model_path = "artifacts/best_model.joblib"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        st.download_button("Download trained model (.joblib)", f, "best_model.joblib")