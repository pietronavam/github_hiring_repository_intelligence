# GitHub Hiring Repository Intelligence

**Track A — Hiring-Oriented Repository Intelligence**

## What does this project do?

This system analyzes GitHub repositories and classifies them by engineering maturity level using a weak-supervision NLP pipeline. It combines GitHub API signals, LLM-generated weak labels (DeepSeek), and a fine-tuned DistilBERT classifier to estimate the seniority level reflected by a repository — without reading the code directly.

## Which track was selected?

**Track A — Hiring-Oriented Repository Intelligence**

The system evaluates whether a repository reflects work expected from an intern, junior, senior, or lead-level engineer, or whether it is a boilerplate template or a low-value abandoned project.

## What repositories were analyzed?

**350 public GitHub repositories** collected via the GitHub REST API using stratified sampling across 5 star tiers to avoid biasing the dataset toward popular repos:

| Tier | Stars range | Expected categories |
|------|-------------|---------------------|
| 0 | 0–2 stars | low_value, intern |
| 1 | 3–50 stars | intern, junior |
| 2 | 51–500 stars | junior, senior |
| 3 | 501+ stars | senior, lead |
| 4 | is:template | template |

The dataset spans **21 programming languages** (top: unknown/no-language, Python, TypeScript, JavaScript, HTML). Star-tier sampling ensures the model sees the full maturity spectrum and does not conflate popularity with engineering quality.

**Label distribution after LLM labeling:**

| Label | Count | % |
|-------|-------|---|
| senior | 97 | 27.7% |
| junior | 92 | 26.3% |
| lead | 62 | 17.7% |
| low_value | 60 | 17.1% |
| intern | 30 | 8.6% |
| template | 9 | 2.6% |

## Which GitHub signals were used?

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

A composite **engineering_score** (0–11) weights CI×3, tests×2, releases×2, contributors>1×2, license×1, readme>1000 chars×1.

## How were repository summaries created?

Each repository is converted into a **natural language summary** combining the 12 numerical/boolean signals into a readable text fed to the LLM:

```text
Repository has 15 contributors, active CI/CD workflows, weekly commits,
regular releases (8 total), long README (3200 chars), 45 stars, 12 forks,
2 years old, last push 3 days ago. Engineering score: 10/11.
```

Text representations allow the LLM to reason about combinations of signals the way a recruiter would, and allow DistilBERT to learn from contextual patterns rather than raw numeric vectors.

## How were prompts designed?

**System prompt** provides an explicit rubric for each category so that labels are consistent across calls. **Few-shot examples** calibrate the output format and tone. **Low temperature (0.1)** reduces label variance. A **confidence field** in the JSON output allows filtering out ambiguous cases (threshold: ≥ 0.6) before training.

```text
System: "You are an expert technical recruiter. Classify the repository
by engineering maturity. Categories: intern, junior, senior, lead,
template, low_value. Return JSON: {label, confidence, reason}"

User: Repository has 45 contributors, active CI/CD, weekly commits,
28 releases, long README, engineering score: 11/11...
```

LLM used: **DeepSeek** (via API, temperature=0.1, JSON output mode).

## How was the dataset split?

| Split | Size | % |
|-------|------|---|
| Train | ~245 | 70% |
| Validation | ~53 | 15% |
| Test | ~53 | 15% |

Split is stratified by label. The test set was held out during all training and only used for final evaluation.

## Which BERT model was used?

**DistilBERT** (`distilbert-base-uncased`) fine-tuned via HuggingFace Transformers on repository text summaries. DistilBERT was chosen over full BERT for computational efficiency given the small dataset size (350 repos).

Training: 5 epochs, batch size 16, AdamW optimizer, warmup steps, CPU training.

## What were the final metrics?

### DistilBERT (fine-tuned)
| Metric | Score |
|--------|-------|
| Accuracy | **56.6%** |
| Precision (weighted) | **42.4%** |
| Recall (weighted) | **56.6%** |
| F1-score (weighted) | **47.5%** |

### Baseline comparison
| Model | Accuracy | F1 (weighted) |
|-------|----------|----------------|
| Majority baseline | 28.3% | 12.5% |
| TF-IDF + Logistic Regression | 66.0% | 61.9% |
| **DistilBERT (fine-tuned)** | **56.6%** | **47.5%** |

DistilBERT underperforms TF-IDF on this dataset. This is expected given the small dataset size (350 repos with weak labels) and the class imbalance (template: 9 samples). TF-IDF exploits the structured token patterns in the `bert_input` field more efficiently when training data is scarce. With a larger, cleaner labeled dataset (1000+ repos), BERT-based models would be expected to outperform TF-IDF.

### Per-class performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| intern | 0.00 | 0.00 | 0.00 |
| junior | 0.64 | 0.54 | 0.58 |
| lead | 0.00 | 0.00 | 0.00 |
| low_value | 0.53 | 0.82 | 0.64 |
| senior | 0.56 | 0.93 | 0.70 |
| template | 0.00 | 0.00 | 0.00 |

Strongest classes: `senior` and `low_value`. Weakest: `intern`, `lead`, `template` (underrepresented in training data).

## What are the main limitations?

- **Label noise**: LLM is not a perfect annotator; edge cases between intern/junior or lead/senior are subjective
- **Small dataset**: 350 repos is insufficient for BERT to outperform simpler baselines; larger samples would improve generalization
- **Selection bias**: GitHub API search is not a random sample of all repositories
- **Class imbalance**: `template` (9 samples) and `intern` (30 samples) are underrepresented
- **Language bias**: English-dominant repos may score higher on documentation signals
- **Stars ≠ quality**: viral or template repos distort star-based sampling

## What are the possible business applications?

- **Technical recruiting**: automate initial screening of candidate portfolios
- **Engineering managers**: quickly assess quality signals across a team's public repos
- **Accelerators / VCs**: evaluate technical depth of startup repos at scale
- **Technical interview pipelines**: use as a pre-screening signal before code interviews
- **Ethical consideration**: this system should never be the sole decision criterion — it estimates repository signals, not developer ability. A strong engineer may have weak public repos due to NDA constraints, private work, or personal choices. It must be used as one signal among many.

## How to run the project?

```bash
pip install -r requirements.txt

# 1. Collect data
python src/github_collector.py

# 2. Preprocess
python src/preprocessing.py

# 3. Generate summaries
python src/summarization.py

# 4. LLM weak labeling
python src/llm_labeling.py

# 5. Train BERT
python src/train.py

# 6. Evaluate
python src/evaluation.py
```

Or run the full pipeline in one command:

```bash
python run_pipeline.py
```

Requires a GitHub token (`GITHUB_TOKEN`) and a DeepSeek API key (`DEEPSEEK_API_KEY`) in a `.env` file.

## How to run the Streamlit app?

```bash
streamlit run app.py
```

The app displays 4 tabs: Problem & Methodology, Exploratory Analysis, Model Results, and Interactive Repository Explorer (including live prediction for any public GitHub repo).
