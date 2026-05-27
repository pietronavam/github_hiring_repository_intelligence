# GitHub Hiring Repository Intelligence

**Track A — Hiring-Oriented Repository Intelligence**

## What does this project do?

This system analyzes GitHub repositories and classifies them by engineering maturity level using a weak-supervision NLP pipeline. It combines GitHub API signals, LLM-generated weak labels, and a fine-tuned BERT classifier to estimate the seniority level reflected by a repository.

## Track Selected

Track A — Hiring-Oriented Repository Intelligence

## Categories

| Label | Description |
|-------|-------------|
| `intern` | Simple scripts, minimal structure, no CI/CD |
| `junior` | Basic project structure, some tests, limited documentation |
| `senior` | Clean architecture, CI/CD, good documentation, test coverage |
| `lead` | Complex systems, multi-service, advanced patterns, release management |
| `template` | Boilerplate or replica repositories with no original work |
| `low_value` | Abandoned, empty, or trivial repositories |

## Repository Signals Used

- Number of contributors
- Commit frequency (last 30/90 days)
- Stars and forks count
- Open/closed issues ratio
- Pull request activity
- Release frequency
- README length and structure
- CI/CD workflow presence
- Repository age and last activity date
- Topics and tags
- Dependency files presence (requirements.txt, package.json, etc.)

## How Repository Summaries Were Created

Each repository is converted into a structured text summary combining numerical signals and qualitative features, fed to an LLM for classification.

## Prompt Design

Prompts describe the repository signals and ask the LLM to classify the engineering maturity level based on defined rubrics for each category.

## Dataset Split

- 70% Training
- 15% Validation  
- 15% Test

## BERT Model Used

DistilBERT (`distilbert-base-uncased`) fine-tuned on repository text summaries.

## How to Run

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

## How to Run the Streamlit App

```bash
streamlit run app.py
```

## Limitations

- LLM labels are noisy by nature (weak supervision)
- GitHub API rate limits restrict dataset size
- Engineering maturity is subjective and context-dependent
- Model performance depends heavily on prompt quality

## Business Applications

- Technical screening for recruiters and hiring managers
- Portfolio evaluation for accelerators and investors
- Automated code quality signals for engineering teams
