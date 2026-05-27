"""
Repository Summarization

Analytical justification for text-based representation:
  BERT models operate on text. Rather than feeding raw numerical features
  directly (which would require a tabular model), we convert each repo's
  signals into a natural language description. This lets the LLM and BERT
  model leverage semantic understanding — for example, the phrase
  "active CI/CD workflows" carries meaning beyond a binary flag.

  We generate two representations per repo:
    1. llm_summary — verbose, used as prompt input for LLM labeling
    2. bert_input  — concise, used as the text input for BERT fine-tuning
"""

import pandas as pd
from pathlib import Path
from src.utils import setup_logging

log = setup_logging("summarization")

PROCESSED_PATH = Path("data/processed/repositories_clean.csv")
SUMMARIZED_PATH = Path("data/processed/repositories_summarized.csv")


def build_llm_summary(row: pd.Series) -> str:
    age = f"{row['repo_age_years']:.1f} years old"
    activity = "recently active" if row["is_recently_active"] else f"last pushed {int(row['days_since_push'])} days ago"
    ci = "has active CI/CD workflows" if row["has_ci"] else "no CI/CD workflows found"
    tests = "has a test directory" if row["has_tests"] else "no test directory detected"
    license_ = "has a license" if row["has_license"] else "no license"
    releases = f"{int(row['releases'])} releases" if row["has_releases"] else "no releases"
    readme = (
        "extensive README (>2000 chars)" if row["readme_length"] > 2000
        else "moderate README (500-2000 chars)" if row["readme_length"] > 500
        else "minimal or no README"
    )
    topics = f"tagged with {int(row['topics_count'])} topics" if row["topics_count"] > 0 else "no topics"
    contribs = int(row["contributors"])
    contrib_str = (
        "single contributor" if contribs <= 1
        else f"2-5 contributors" if contribs <= 5
        else f"6-20 contributors" if contribs <= 20
        else f"20+ contributors"
    )
    commits = int(row["commits_30d"])
    commit_str = (
        "no commits in the last 30 days" if commits == 0
        else f"{commits} commits in the last 30 days"
    )

    return (
        f"Repository: {row['full_name']}\n"
        f"Language: {row.get('language', 'unknown')}\n"
        f"Description: {row.get('description', 'none')}\n"
        f"Age: {age}, {activity}\n"
        f"Stars: {int(row['stars'])}, Forks: {int(row['forks'])}, Open issues: {int(row['open_issues'])}\n"
        f"Contributors: {contrib_str}\n"
        f"Activity: {commit_str}\n"
        f"CI/CD: {ci}\n"
        f"Testing: {tests}\n"
        f"Documentation: {readme}\n"
        f"Releases: {releases}\n"
        f"License: {license_}\n"
        f"Topics: {topics}\n"
        f"Size: {int(row['size_kb'])} KB\n"
        f"Engineering score (composite): {int(row['engineering_score'])}/11"
    )


def build_bert_input(row: pd.Series) -> str:
    ci = "ci-yes" if row["has_ci"] else "ci-no"
    tests = "tests-yes" if row["has_tests"] else "tests-no"
    license_ = "license-yes" if row["has_license"] else "license-no"
    releases = "releases-yes" if row["has_releases"] else "releases-no"
    active = "active" if row["is_recently_active"] else "inactive"
    contribs = int(row["contributors"])
    contrib_bucket = (
        "solo" if contribs <= 1
        else "small-team" if contribs <= 5
        else "mid-team" if contribs <= 20
        else "large-team"
    )
    star_bucket = (
        "no-stars" if row["stars"] == 0
        else "low-stars" if row["stars"] < 10
        else "mid-stars" if row["stars"] < 200
        else "high-stars"
    )
    readme_bucket = (
        "no-readme" if row["readme_length"] < 100
        else "short-readme" if row["readme_length"] < 500
        else "long-readme"
    )

    return (
        f"lang:{row.get('language','unknown').lower()} "
        f"{ci} {tests} {license_} {releases} {active} "
        f"{contrib_bucket} {star_bucket} {readme_bucket} "
        f"score:{int(row['engineering_score'])} "
        f"age:{int(row['repo_age_years'])}y "
        f"commits30d:{int(row['commits_30d'])} "
        f"{row.get('description', '')[:120]}"
    )


def summarize(input_path: Path = PROCESSED_PATH, output_path: Path = SUMMARIZED_PATH) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    log.info(f"Building summaries for {len(df)} repos")
    df["llm_summary"] = df.apply(build_llm_summary, axis=1)
    df["bert_input"] = df.apply(build_bert_input, axis=1)
    df.to_csv(output_path, index=False)
    log.info(f"Summaries saved → {output_path}")
    return df


if __name__ == "__main__":
    summarize()
