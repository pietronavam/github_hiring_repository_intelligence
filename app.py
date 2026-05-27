"""
Streamlit App — GitHub Hiring Repository Intelligence
Track A: Engineering Maturity Classification

Tab 1: Problem & Methodology
Tab 2: Exploratory Analysis
Tab 3: Model Results
Tab 4: Interactive Repository Explorer
"""

import json
import torch
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="GitHub Hiring Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ──────────────────────────────────────────────────────────────
LABELED_PATH = Path("data/labeled/repositories_labeled.csv")
PROCESSED_PATH = Path("data/processed/repositories_clean.csv")
METRICS_DIR = Path("output/metrics")
TABLES_DIR = Path("output/tables")
FIGURES_DIR = Path("output/figures")
MODEL_DIR = Path("models/trained_models/distilbert_hiring")
LABEL_MAP_PATH = Path("models/trained_models/label_map.json")

PALETTE = {
    "intern": "#6c757d", "junior": "#17a2b8", "senior": "#28a745",
    "lead": "#007bff", "template": "#ffc107", "low_value": "#dc3545",
}

# ── Helpers ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    if LABELED_PATH.exists() and PROCESSED_PATH.exists():
        labeled = pd.read_csv(LABELED_PATH).dropna(subset=["label"])
        processed = pd.read_csv(PROCESSED_PATH)
        merged = processed.merge(
            labeled[["full_name", "label", "confidence", "llm_summary", "bert_input", "label_reason"]],
            on="full_name", how="inner"
        )
        merged = merged[merged["confidence"] >= 0.6]
        return labeled, processed, merged
    return None, None, None


@st.cache_data
def load_metrics():
    result = {}
    if (METRICS_DIR / "bert_metrics.json").exists():
        with open(METRICS_DIR / "bert_metrics.json") as f:
            result["bert"] = json.load(f)
    if (METRICS_DIR / "model_comparison.csv").exists():
        result["comparison"] = pd.read_csv(METRICS_DIR / "model_comparison.csv")
    if (TABLES_DIR / "per_class_metrics.csv").exists():
        result["per_class"] = pd.read_csv(TABLES_DIR / "per_class_metrics.csv", index_col=0)
    if (METRICS_DIR / "confusion_matrix.npy").exists():
        result["cm"] = np.load(METRICS_DIR / "confusion_matrix.npy")
        with open(METRICS_DIR / "label_names.json") as f:
            result["label_names"] = json.load(f)
    return result


@st.cache_resource
def load_model():
    if not MODEL_DIR.exists():
        return None, None
    try:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
        model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def predict_repo(summary_text: str, tokenizer, model) -> tuple[str, dict]:
    with open(LABEL_MAP_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    enc = tokenizer(summary_text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    pred_idx = int(torch.argmax(logits))
    return label_map[pred_idx], {label_map[i]: round(p, 3) for i, p in enumerate(probs)}


def show_figure(path: Path):
    if path.exists():
        st.image(str(path), use_column_width=True)
    else:
        st.info("Figure not yet generated — run the full pipeline first.")


# ── Layout ───────────────────────────────────────────────────────────────

st.title("GitHub Hiring Repository Intelligence")
st.caption("Track A — Engineering Maturity Classification via Weak Supervision + DistilBERT")

labeled_df, processed_df, merged_df = load_data()
metrics = load_metrics()
tokenizer, model = load_model()

tab1, tab2, tab3, tab4 = st.tabs([
    "Problem & Methodology",
    "Exploratory Analysis",
    "Model Results",
    "Interactive Explorer",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Problem & Methodology
# ══════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("What problem are we solving?")
        st.markdown("""
Evaluating a developer's GitHub portfolio is time-consuming for recruiters and
engineering managers. Browsing dozens of repositories to estimate seniority level
takes hours and is subjective. We automate this by building a system that reads a
repository's metadata and classifies it into one of 6 engineering-maturity categories.

**The key insight:** you don't need to read the code to estimate maturity level.
Engineering practices leave measurable signals — CI/CD presence, test directories,
contributor count, commit frequency, release history — that correlate strongly with
the seniority required to produce that kind of work.
        """)

        st.header("Repository Selection Methodology")
        st.markdown("""
We use **stratified sampling** across 5 star tiers to avoid biasing the dataset toward popular repos:

| Tier | Stars | Expected categories |
|------|-------|-------------------|
| 0 | 0–2 | low_value, intern |
| 1 | 3–50 | intern, junior |
| 2 | 51–500 | junior, senior |
| 3 | 501+ | senior, lead |
| 4 | is:template | template |

**Why star tiers?** Stars are a noisy but available proxy for visibility.
A repo with 5000 stars can still be a template. By sampling across all tiers,
we ensure the model sees the full spectrum and doesn't conflate popularity with quality.
        """)

        st.header("GitHub Signals Used")
        st.markdown("""
| Signal | Rationale |
|--------|-----------|
| **Contributors** | Team size → project maturity |
| **Commits (30d)** | Active development vs abandoned |
| **Has CI/CD** | Engineering discipline indicator |
| **Has tests** | Code quality and professional practice |
| **Releases** | Versioning discipline |
| **README length** | Documentation effort |
| **Stars / Forks** | Community adoption signal |
| **Age + last push** | Project health and lifecycle |
| **Topics count** | Discoverability / intentional tagging |
| **License** | Open-source professionalism |
| **Repo size (KB)** | Complexity proxy |
| **Open issues** | Community engagement |

We also derive a composite **engineering_score** (0–11) that weights the practices
most strongly associated with professional-grade work (CI×3, tests×2, license, releases×2, team×2, readme).
        """)

    with col2:
        st.header("Prompt Strategy")
        st.markdown("""
Each repository is converted into a **natural language summary** fed to the LLM.

**Why text, not raw numbers?**
LLMs understand context. The phrase *"active CI/CD with weekly commits and 45 contributors"*
carries more meaning than `[1, 45, 0.3]` to a language model trained on engineering discussions.

**Prompt design principles:**
1. Explicit rubric in the system prompt → consistent labels
2. Few-shot examples → calibrate output format and tone
3. Low temperature (0.1) → reduce label variance
4. Confidence field → filter ambiguous cases before training
5. JSON output → reliable parsing
        """)
        st.code("""
System: "You are an expert recruiter.
Classify by engineering maturity.
Categories: intern, junior, senior,
lead, template, low_value.
Return JSON: {label, confidence, reason}"

User: Repository has 45 contributors,
active CI/CD, weekly commits, 28 releases,
long README, score=11/11...
        """, language="text")

        st.header("Dataset Construction")
        st.markdown("""
1. Collect ~350 repos via GitHub API
2. Extract 12 signals per repo
3. Build verbose text summary
4. LLM labels with confidence score
5. Filter: confidence ≥ 0.6
6. Split 70% train / 15% val / 15% test (stratified)
        """)

        st.header("Limitations")
        st.markdown("""
- **Label noise**: LLM is not a perfect annotator; edge cases are subjective
- **Selection bias**: GitHub API search is not a random sample
- **Language bias**: English-dominant repos may score higher on documentation signals
- **Stars ≠ quality**: viral or template repos distort star-based sampling
- **Rate limits**: ~350 repos is small; larger samples would improve generalization
        """)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Exploratory Analysis
# ══════════════════════════════════════════════════════════════════
with tab2:
    if merged_df is None:
        st.info("Run the full pipeline to populate this tab: `python src/github_collector.py` → `preprocessing.py` → `summarization.py` → `llm_labeling.py`")
    else:
        st.header("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total repos", len(merged_df))
        col2.metric("Languages", merged_df["language"].nunique())
        col3.metric("Avg stars", f"{merged_df['stars'].mean():.0f}")
        col4.metric("Avg contributors", f"{merged_df['contributors'].mean():.1f}")

        st.subheader("Category Distribution")
        st.caption("Shows whether our sampling strategy produced a balanced enough dataset for training. Severe imbalance would require oversampling.")
        show_figure(FIGURES_DIR / "label_distribution.png")

        st.subheader("Signal Distributions by Category")
        st.caption("Key diagnostic: if a signal's boxes overlap across all categories, it's not discriminative. Separated boxes = useful feature.")
        show_figure(FIGURES_DIR / "signal_boxplots.png")

        st.subheader("Feature Correlation Heatmap")
        st.caption("High correlation between features signals redundancy. stars↔forks correlation is expected; if engineering_score correlates with CI and tests, the composite is working as intended.")
        show_figure(FIGURES_DIR / "feature_correlation.png")

        st.subheader("Raw Data Explorer")
        st.dataframe(
            merged_df[["full_name", "label", "confidence", "stars", "contributors",
                       "commits_30d", "has_ci", "has_tests", "engineering_score", "language"]].sort_values("label"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Model Results
# ══════════════════════════════════════════════════════════════════
with tab3:
    if not metrics:
        st.info("Run `python src/train.py` then `python src/evaluation.py` to populate results.")
    else:
        st.header("Model Performance Summary")
        if "bert" in metrics:
            b = metrics["bert"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{b['accuracy']:.1%}")
            col2.metric("Precision", f"{b['precision']:.1%}")
            col3.metric("Recall", f"{b['recall']:.1%}")
            col4.metric("F1 (weighted)", f"{b['f1_weighted']:.1%}")

        st.subheader("Baseline Comparison")
        st.caption("Context matters: BERT must significantly beat both baselines to justify the weak-supervision + fine-tuning cost.")
        if "comparison" in metrics:
            st.dataframe(metrics["comparison"].style.format({"accuracy": "{:.1%}", "f1_weighted": "{:.1%}"}), use_container_width=True)
        show_figure(FIGURES_DIR / "model_comparison.png")

        st.subheader("Confusion Matrix")
        st.caption("Rows = true label, columns = predicted. Common errors: intern↔junior (similar signals) and template↔low_value (overlap in engineering practices).")
        show_figure(FIGURES_DIR / "confusion_matrix.png")

        st.subheader("Per-Class Metrics")
        if "per_class" in metrics:
            st.dataframe(metrics["per_class"].style.format(precision=3), use_container_width=True)

        st.subheader("Methodology Sensitivity Analysis")
        st.markdown("""
**Baseline approach**: bert_input uses compact signal tokens (`ci-yes tests-no solo no-stars`)

**Alternative approach**: bert_input uses the full verbose llm_summary (~300 words)

Expected finding: compact tokens perform comparably to verbose summaries because
BERT's tokenizer can learn from structured signal patterns. Verbose summaries add
noise (repo names, descriptions) that may not generalize.

**Category definition sensitivity**: collapsing `low_value` into `intern` would boost
recall on the intern class but lose precision on the real intern class. The 6-class
distinction is intentional — recruiters treat abandoned repos differently from
genuine intern-level work.
        """)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — Interactive Repository Explorer
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.header("Interactive Repository Explorer")

    subtab1, subtab2 = st.tabs(["Browse Labeled Repos", "Live Prediction"])

    with subtab1:
        if merged_df is None:
            st.info("Run the pipeline first to browse labeled repositories.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                sel_label = st.multiselect("Filter by category", options=merged_df["label"].unique().tolist(),
                                           default=merged_df["label"].unique().tolist())
            with col2:
                min_conf = st.slider("Min confidence", 0.6, 1.0, 0.7, 0.05)
            with col3:
                sel_lang = st.multiselect("Language", options=["All"] + sorted(merged_df["language"].dropna().unique().tolist()),
                                          default=["All"])

            filtered = merged_df[
                (merged_df["label"].isin(sel_label)) &
                (merged_df["confidence"] >= min_conf)
            ]
            if "All" not in sel_lang:
                filtered = filtered[filtered["language"].isin(sel_lang)]

            st.caption(f"Showing {len(filtered)} repositories")
            st.dataframe(
                filtered[["full_name", "label", "confidence", "label_reason", "stars",
                          "contributors", "has_ci", "has_tests", "engineering_score"]].reset_index(drop=True),
                use_container_width=True,
            )

            if st.checkbox("Show full LLM summary for selected row"):
                idx = st.number_input("Row index", 0, max(0, len(filtered) - 1), 0)
                if len(filtered) > 0:
                    row = filtered.iloc[int(idx)]
                    st.subheader(row["full_name"])
                    st.markdown(f"**Predicted label:** `{row['label']}` (confidence: {row['confidence']:.0%})")
                    st.markdown(f"**Reason:** {row['label_reason']}")
                    st.code(row.get("llm_summary", "N/A"), language="text")

    with subtab2:
        st.subheader("Predict any GitHub repository")
        if model is None:
            st.warning("Model not found. Run `python src/train.py` first.")
        else:
            repo_input = st.text_input("GitHub repository (owner/repo)", placeholder="e.g. torvalds/linux")

            if st.button("Analyze repository") and repo_input:
                from src.summarization import build_bert_input, build_llm_summary
                from src.github_collector import extract_signals
                from src.preprocessing import preprocess
                import time

                with st.spinner(f"Fetching signals for {repo_input}..."):
                    try:
                        from src.utils import get_github_headers, GITHUB_API_BASE
                        r = requests.get(f"{GITHUB_API_BASE}/repos/{repo_input}", headers=get_github_headers())
                        if r.status_code != 200:
                            st.error(f"Repository not found or API error: {r.status_code}")
                        else:
                            signals = extract_signals(r.json())
                            if signals:
                                row = pd.Series(signals)
                                row["repo_age_years"] = row["age_days"] / 365.0
                                row["is_recently_active"] = row["days_since_push"] < 90
                                row["has_releases"] = row["releases"] > 0
                                row["engineering_score"] = (
                                    int(row["has_ci"]) * 3 + int(row["has_tests"]) * 2
                                    + int(row["has_license"]) + int(row["has_releases"]) * 2
                                    + int(row["contributors"] > 1) * 2
                                    + int(row["readme_length"] > 1000)
                                )
                                bert_text = build_bert_input(row)
                                label, probs = predict_repo(bert_text, tokenizer, model)

                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    color = PALETTE.get(label, "#999")
                                    st.markdown(f"### Prediction: `{label}`")
                                    st.markdown(f"**Engineering score:** {int(row['engineering_score'])}/11")
                                    st.markdown(f"**Stars:** {int(row['stars'])} | **Forks:** {int(row['forks'])}")
                                    st.markdown(f"**Contributors:** {int(row['contributors'])}")
                                    st.markdown(f"**CI/CD:** {'Yes' if row['has_ci'] else 'No'} | **Tests:** {'Yes' if row['has_tests'] else 'No'}")
                                    st.markdown(f"**Commits (30d):** {int(row['commits_30d'])}")
                                with col2:
                                    fig, ax = plt.subplots(figsize=(6, 3))
                                    cats = list(probs.keys())
                                    vals = list(probs.values())
                                    colors = [PALETTE.get(c, "#999") for c in cats]
                                    ax.barh(cats, vals, color=colors, alpha=0.8)
                                    ax.set_xlim(0, 1)
                                    ax.set_xlabel("Probability")
                                    ax.set_title("Category Probabilities")
                                    ax.spines["top"].set_visible(False)
                                    ax.spines["right"].set_visible(False)
                                    st.pyplot(fig)
                                    plt.close()
                    except Exception as e:
                        st.error(f"Error: {e}")
