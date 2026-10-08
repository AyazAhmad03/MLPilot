"""MLPilot - Streamlit interface.

Run from the project root:  streamlit run app.py
Keep .streamlit/config.toml in the folder you run the command from.
Requires: streamlit, pandas, numpy, scikit-learn, joblib
"""
import html
import io
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLPilot", page_icon="🧭", layout="wide")

# The function that builds and compiles your LangGraph.
# If it now lives in another module, change this one import.
try:
    from MLPilot import create_graph
    IMPORT_ERROR = None
except Exception as exc:
    create_graph, IMPORT_ERROR = None, exc

STEPS = {
    "data_upload": ("Data upload", "Upload"),
    "data_analysis": ("Data analysis", "Analysis"),
    "problem_detection": ("Problem detection", "Detection"),
    "data_preparation": ("Data preparation", "Preparation"),
    "model_selection": ("Model selection", "Selection"),
    "model_experimentation": ("Model experimentation", "Experiments"),
    "training": ("Training", "Training"),
    "evaluation": ("Evaluation", "Evaluation"),
    "report": ("Report", "Report"),
}
ORDER = list(STEPS)

# --------------------------------------------------------------------------
# Styling and animation
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
:root{--bg:#130F26;--panel:#1B1636;--line:#322A58;--text:#ECE8F7;--muted:#9A92BD;--amber:#F3B562;--mint:#7FD6B2;--rose:#E88BA0;--violet:#6C5FB5}
.stApp{font-family:'Instrument Sans',system-ui,sans-serif;background:radial-gradient(900px 420px at 88% -10%,#261D4D 0%,transparent 65%),var(--bg)}
h1,h2,h3,.ht,.sec,.cv{font-family:'Bricolage Grotesque',sans-serif!important;letter-spacing:-.01em}
#MainMenu,footer,.stDeployButton,[data-testid="stToolbar"]{display:none!important}
header[data-testid="stHeader"]{background:transparent;height:0}
.block-container{padding-top:2rem;max-width:1180px}
section[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line)}

/* Header banner */
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:18px;padding:22px 28px 16px;
  background:linear-gradient(120deg,#231B4A,#181331 55%,#201840);background-size:200% 200%;animation:drift 18s ease-in-out infinite}
.hero::after{content:"";position:absolute;right:-60px;top:-90px;width:300px;height:300px;border-radius:50%;
  background:radial-gradient(circle,rgba(243,181,98,.16),transparent 65%);pointer-events:none}
@keyframes drift{50%{background-position:100% 50%}}
.hrow{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.hl{display:flex;align-items:center;gap:14px}
.mark{width:44px;height:44px;border-radius:13px;display:grid;place-items:center;font:700 17px 'Bricolage Grotesque',sans-serif;
  color:#2A1A05;background:linear-gradient(135deg,#F7C77E,#E08F3E);box-shadow:0 6px 22px rgba(243,181,98,.28)}
.ht{font-size:26px;font-weight:700;line-height:1.1}
.hs{color:var(--muted);font-size:14px;margin-top:2px}
.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:5px 14px;font-size:13px;color:var(--muted);background:rgba(19,15,38,.6)}
.pill i{width:8px;height:8px;border-radius:50%;background:var(--muted);display:inline-block}
.pill.fly,.pill.done,.pill.err{color:var(--text)}
.pill.fly i{background:var(--amber);animation:blink 1s ease-in-out infinite}
.pill.done i{background:var(--mint)} .pill.err i{background:var(--rose)}
@keyframes blink{50%{opacity:.25}}

.track{position:relative;display:flex;margin:26px 0 8px}
.st{position:relative;flex:1;text-align:center}
.st::before{content:"";position:absolute;top:6px;left:-50%;width:100%;height:2px;background:var(--line);transition:background .6s}
.st:first-child::before{display:none}
.dot{position:relative;z-index:1;display:block;width:14px;height:14px;margin:0 auto;border-radius:50%;background:var(--bg);border:2px solid var(--line);transition:all .5s}
.st em{display:block;font-style:normal;font-size:11.5px;color:var(--muted);margin-top:8px;transition:color .4s}
.st.done::before{background:var(--mint)} .st.done .dot{background:var(--mint);border-color:var(--mint)} .st.done em{color:var(--text)}
.st.on::before{background:linear-gradient(90deg,var(--mint),var(--amber))}
.st.on .dot{border-color:var(--amber);background:var(--amber);animation:ring 1.3s ease-out infinite} .st.on em{color:var(--amber);font-weight:600}
@keyframes ring{0%{box-shadow:0 0 0 0 rgba(243,181,98,.55)}100%{box-shadow:0 0 0 12px rgba(243,181,98,0)}}
.traveler{position:absolute;top:3px;left:0;width:20px;height:8px;border-radius:6px;background:var(--amber);filter:blur(5px);opacity:.9;animation:fly 6s ease-in-out infinite}
@keyframes fly{0%{left:3%;opacity:0}12%{opacity:.9}88%{opacity:.9}100%{left:96%;opacity:0}}
.note{min-height:22px;font-size:14px;color:var(--muted);margin-top:6px}
.note b{color:var(--text);font-weight:600}
@media(max-width:760px){.st em{display:none}.hero{padding:18px}}

/* Cards and results */
.sec{font-size:19px;font-weight:700;margin:30px 0 4px}
.hint{color:var(--muted);font-size:14px;margin:0 0 14px}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.card span{display:block;color:var(--muted);font-size:13px}
.card .cv{display:block;font-size:26px;font-weight:700;margin-top:4px;word-break:break-word}
.rise{animation:rise .55s cubic-bezier(.2,.8,.2,1) both;animation-delay:var(--d,0s)}
@keyframes rise{from{opacity:0;transform:translateY(10px)}}
.bars{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:8px 20px}
.brow{display:grid;grid-template-columns:minmax(120px,190px) 1fr 64px;gap:14px;align-items:center;padding:11px 0;font-size:14px}
.brow+.brow{border-top:1px solid rgba(50,42,88,.6)}
.bt{height:10px;border-radius:6px;background:var(--bg);overflow:hidden}
.bf{height:100%;width:var(--w);border-radius:6px;background:var(--violet);animation:grow 1.1s cubic-bezier(.2,.8,.2,1) both;animation-delay:var(--d,0s)}
@keyframes grow{from{width:0}}
.brow b{text-align:right;font-weight:600}
.brow.best .bf{background:linear-gradient(90deg,#E08F3E,var(--amber));box-shadow:0 0 14px rgba(243,181,98,.45)}
.brow.best b,.brow.best .bn{color:var(--amber)} .brow.best .bn{font-weight:600}
@media(max-width:560px){.brow{grid-template-columns:100px 1fr 54px;gap:10px}}

/* Streamlit widgets */
button[kind="primary"],[data-testid="stBaseButton-primary"]{background:var(--amber)!important;border-color:var(--amber)!important;color:#2A1A05!important;font-weight:600;transition:transform .15s,filter .15s}
button[kind="primary"]:hover,[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.08);transform:translateY(-1px)}
[data-baseweb="select"]>div:focus-within,[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--amber)!important}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:var(--panel)}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------
@st.cache_data
def load_iris() -> pd.DataFrame:
    if Path("iris.csv").exists():
        return pd.read_csv("iris.csv")
    from sklearn.datasets import load_iris as _load

    data = _load(as_frame=True)
    out = data.frame.copy()
    out["species"] = out.pop("target").map(dict(enumerate(data.target_names)))
    return out


@st.cache_data
def make_churn(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    tenure = rng.integers(1, 73, n)
    charges = rng.uniform(20, 110, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    internet = rng.choice(["Fiber", "DSL", "None"], n)
    calls = np.minimum(rng.poisson(1.6, n), 9)
    payment = rng.choice(["Card", "Bank transfer", "Cheque", "Wallet"], n)
    z = (-1 + np.where(contract == "Month-to-month", 1.3, -.6) + np.where(internet == "Fiber", .7, 0)
         + calls * .28 - tenure * .03 + charges * .008)
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    out = pd.DataFrame({"tenure_months": tenure, "monthly_charges": charges, "contract": contract,
                        "internet_service": internet, "support_calls": calls,
                        "payment_method": payment, "Churn": churn})
    out.loc[rng.random(n) < .04, "monthly_charges"] = np.nan
    out.loc[rng.random(n) < .03, "payment_method"] = np.nan
    return out


SAMPLES = {"Iris flowers": load_iris, "Customer churn (synthetic)": make_churn}


def guess_target(columns) -> int:
    lowered = [c.lower() for c in columns]
    for key in ("churn", "species", "target", "label", "class", "outcome"):
        if key in lowered:
            return lowered.index(key)
    return len(columns) - 1


# --------------------------------------------------------------------------
# HTML builders (each returns one block with no blank lines)
# --------------------------------------------------------------------------
LABELS = {"f1_score": "F1 score", "r2_score": "R² score", "mae": "MAE", "rmse": "RMSE"}


def header_html(kind: str, label: str, done: int = 0, active=None, note: str = "") -> str:
    nodes = []
    for i, key in enumerate(ORDER):
        state = "done" if i < done else ("on" if i == active else "")
        nodes.append(f'<div class="st {state}"><span class="dot"></span><em>{STEPS[key][1]}</em></div>')
    traveler = '<div class="traveler"></div>' if done == 0 and active is None and kind == "" else ""
    return (
        '<div class="hero"><div class="hrow"><div class="hl"><div class="mark">ML</div>'
        '<div><div class="ht">MLPilot</div><div class="hs">Automated machine learning pipeline</div></div></div>'
        f'<span class="pill {kind}"><i></i>{label}</span></div>'
        f'<div class="track">{"".join(nodes)}{traveler}</div><div class="note">{note}</div></div>'
    )


def cards_html(items, start_delay: float = 0.0, animate: bool = True) -> str:
    out = []
    for i, (label, value) in enumerate(items):
        cls = "card rise" if animate else "card"
        out.append(f'<div class="{cls}" style="--d:{start_delay + i * 0.08:.2f}s"><span>{html.escape(str(label))}</span>'
                   f'<span class="cv">{html.escape(str(value))}</span></div>')
    return f'<div class="g">{"".join(out)}</div>'


def bars_html(cv: dict, best: str) -> str:
    rows = []
    for i, (name, score) in enumerate(sorted(((k, v["mean_score"]) for k, v in cv.items()), key=lambda x: -x[1])):
        width = max(0.0, min(100.0, score * 100))
        cls = "brow best" if name == best else "brow"
        rows.append(f'<div class="{cls}" style="--w:{width:.1f}%;--d:{0.15 + i * 0.12:.2f}s"><span class="bn">{html.escape(name)}</span>'
                    f'<div class="bt"><div class="bf"></div></div><b>{score:.4f}</b></div>')
    return f'<div class="bars">{"".join(rows)}</div>'


def section(title: str, hint: str = "") -> None:
    st.markdown(f'<div class="sec">{title}</div><div class="hint">{hint}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def render_results(res: dict) -> None:
    s = res["state"]
    metrics, cv, best = s.get("metrics", {}), s.get("cv_results", {}), s.get("best_model_name", "")
    ptype = s.get("problem_type", "")
    scoring = "weighted F1" if ptype == "classification" else "R²"

    section("Summary")
    st.markdown(cards_html([("Problem type", ptype.title()), ("Best model", best), ("Models compared", len(cv))]),
                unsafe_allow_html=True)

    section("Test performance", f"{html.escape(best)} scored on the held-out 20% of the data.")
    st.markdown(cards_html([(LABELS.get(k, k.title()), f"{v:.4f}") for k, v in metrics.items()], 0.2),
                unsafe_allow_html=True)

    section("Model comparison", f"Mean 5-fold cross-validation score ({scoring}). The selected model is highlighted.")
    st.markdown(bars_html(cv, best), unsafe_allow_html=True)

    section("Details")
    with st.expander("Report"):
        st.code(s.get("report", "").strip(), language="text")
    with st.expander("Fold-by-fold scores"):
        folds = pd.DataFrame({k: v["scores"] for k, v in cv.items()}).round(4)
        folds.index = [f"Fold {i + 1}" for i in range(len(folds))]
        st.dataframe(folds, use_container_width=True)
    with st.expander("Run log"):
        st.code(res["log"].strip(), language="text")

    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download report", s.get("report", ""), "mlpilot_report.txt", use_container_width=True)
    buf = io.BytesIO()
    joblib.dump(s["trained_pipeline"], buf)
    d2.download_button("Download model", buf.getvalue(), "mlpilot_pipeline.joblib", use_container_width=True)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Setup")
    source = st.selectbox("Data source", ["Sample dataset", "Upload CSV"])
    df, dataset_name = None, ""
    if source == "Sample dataset":
        choice = st.selectbox("Sample", list(SAMPLES))
        df, dataset_name = SAMPLES[choice](), choice
    else:
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            try:
                df, dataset_name = pd.read_csv(up), up.name
            except Exception as exc:
                st.error(f"Could not read that file: {exc}")
    target = None
    if df is not None:
        target = st.selectbox("Target column", list(df.columns), index=guess_target(df.columns))
    launch = st.button("Launch pipeline", type="primary", use_container_width=True,
                       disabled=df is None or create_graph is None)

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

header_slot = st.empty()

if IMPORT_ERROR is not None:
    header_slot.markdown(header_html("err", "Not ready", note="MLPilot.py could not be imported."), unsafe_allow_html=True)
    st.error(f"Could not import `create_graph` from MLPilot.py: {IMPORT_ERROR}. "
             "Change the import at the top of app.py to the module that builds your graph.")
    st.stop()

if df is None:
    header_slot.markdown(header_html("", "Standing by", note="Choose a sample dataset or upload a CSV in the sidebar."),
                         unsafe_allow_html=True)
    st.stop()

section("Dataset", html.escape(dataset_name))
st.markdown(cards_html([
    ("Rows", f"{len(df):,}"),
    ("Columns", df.shape[1]),
    ("Numeric columns", df.select_dtypes(include=np.number).shape[1]),
    ("Columns with gaps", int((df.isnull().sum() > 0).sum())),
], animate=False), unsafe_allow_html=True)
st.write("")
with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)

if launch:
    st.session_state.result = None
    header_slot.markdown(header_html("fly", "Running", 0, 0, f"Running: <b>{STEPS[ORDER[0]][0]}</b>"),
                         unsafe_allow_html=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()
    df.to_csv(tmp.name, index=False)  # the data_upload agent reads from a file path

    final_state, buffer = {}, io.StringIO()
    done = 0
    try:
        graph = create_graph()
        with redirect_stdout(buffer):
            for chunk in graph.stream({"file_path": tmp.name, "target_column": target}, stream_mode="updates"):
                for node, update in chunk.items():
                    final_state.update(update or {})
                    done = ORDER.index(node) + 1 if node in ORDER else done
                    if done < len(ORDER):
                        header_slot.markdown(
                            header_html("fly", "Running", done, done, f"Running: <b>{STEPS[ORDER[done]][0]}</b>"),
                            unsafe_allow_html=True)
        st.session_state.result = {"state": final_state, "log": buffer.getvalue()}
        header_slot.markdown(header_html("done", "Complete", len(ORDER), None, "All nine agents completed."),
                             unsafe_allow_html=True)
        st.toast("Pipeline complete", icon="✅")
    except Exception as exc:
        header_slot.markdown(header_html("err", "Stopped", done, None, "The pipeline stopped before finishing."),
                             unsafe_allow_html=True)
        st.error(f"The pipeline stopped: {exc}")
        with st.expander("Run log"):
            st.code(buffer.getvalue().strip() or "No output.", language="text")
    finally:
        os.unlink(tmp.name)
elif st.session_state.result is not None:
    header_slot.markdown(header_html("done", "Complete", len(ORDER), None, "All nine agents completed."),
                         unsafe_allow_html=True)
else:
    header_slot.markdown(header_html("", "Standing by", note="Pick the target column, then select <b>Launch pipeline</b>."),
                         unsafe_allow_html=True)

if st.session_state.result is not None:
    render_results(st.session_state.result)