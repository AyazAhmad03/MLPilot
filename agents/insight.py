"""
agents/insight.py — the LLM Insight Agent (MLPilot's only LLM node).

Runs once, after the performance gate ends the tuning loop. Converts
the run's raw numbers (metrics, tuning history, cleaning decisions)
into a plain-English verdict: why the model won, what the numbers say
about quality, and one concrete next step.

Design rule: this node can NEVER crash or block the pipeline.
No API key, no internet, provider down -> graceful fallback text.
"""

import os

from dotenv import load_dotenv

from core.state import GraphState

load_dotenv()  # reads .env from the project root if present

# LangChain LCEL chain: prompt | llm | parser
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior ML engineer reviewing a completed AutoML run. "
     "Write a short, grounded analysis for the user. "
     "Use ONLY the facts in the context below — never invent or estimate numbers. "
     "If a fact is not in the context, do not mention it. "
     "Maximum 150 words."),
    ("human",
     "Run context:\n{context}\n\n"
     "Give your analysis in exactly 3 short labelled points:\n"
     "1. Why the winning model won: if candidate CV scores are provided, "
     "compare them and say how clear the win was.\n"
     "2. An honest quality read: flag the most notable signal in the numbers — "
     "especially any gap between cross-validation and test scores, or a small "
     "test set (<30 samples) that makes results fragile. If neither applies, "
     "say the results look stable.\n"
     "3. One concrete next step: the single highest-value action to try next.\n\n"
     "Keep the tone plain and direct."),
])


def _build_context(state: GraphState) -> str:
    """Collect everything the LLM should know about this run."""
    model_name = (state.get("model_name")
                  or state.get("best_model_name")
                  or "unknown")
    lines = [
        f"Problem type: {state.get('problem_type', 'unknown')}",
        f"Winning model: {model_name}",
        f"Metrics: {state.get('metrics', {})}",
        f"Tuning attempted: {state.get('tuning_attempted', False)}",
        f"Tuning history: {state.get('tuning_history', [])}",
        f"Columns dropped during cleaning: {state.get('dropped_columns', [])}",
    ]
    
    cv_results = state.get("cv_results")
    if cv_results:
        scores_text = ", ".join(
            f"{name}: {res['mean_score']:.4f}" for name, res in cv_results.items()
        )
        lines.append(f"Candidate CV scores: {scores_text}")

    X_test = state.get("X_test")
    n_test = len(X_test) if X_test is not None else 0
    if n_test:
        lines.append(f"Test set size: {n_test} samples")
        
    return "\n".join(lines)


def insight_agent(state: GraphState) -> dict:
    context = _build_context(state)

    fallback = (
        "LLM insight unavailable (no API key or provider unreachable). "
        "The pipeline results above are still valid. Set OPENAI_API_KEY "
        "to enable plain-English explanations."
    )

    if not LANGCHAIN_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return {"insight": fallback}

    try:
        kwargs = {"model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                  "temperature": 0.2}
        if os.getenv("LLM_BASE_URL"):
            kwargs["base_url"] = os.getenv("LLM_BASE_URL")  # Groq etc.
        llm = ChatOpenAI(**kwargs)

        chain = PROMPT | llm | StrOutputParser()
        return {"insight": chain.invoke({"context": context}).strip()}
    except Exception as exc:
        return {"insight": f"{fallback} (error: {exc})"}