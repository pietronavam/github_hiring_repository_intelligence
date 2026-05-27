"""
Preprocessing

Analytical choices:
  - Derived signals are more informative than raw counts. A repo with 50 commits
    but created 5 years ago is very different from one created last month.
    We create ratios (commits_per_day, forks_per_star) to capture this.

  - Forks are removed from the training set (is_fork=True) because they reflect
    the original repo's quality, not the forker's work. This is a deliberate
    design decision to avoid label leakage.

  - We keep is_fork repos in a separate file for reference but exclude them
    from modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import setup_logging

log = setup_logging("preprocessing")

RAW_PATH = Path("data/raw/repositories.csv")
PROCESSED_PATH = Path("data/processed/repositories_clean.csv")


def preprocess(input_path: Path = RAW_PATH, output_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df)} repos from {input_path}")

    # Remove forks — they reflect the original repo's maturity, not the forker's
    df = df[~df["is_fork"].astype(bool)].copy()
    log.info(f"{len(df)} repos after removing forks")

    # Fill missing values
    numeric_cols = ["stars", "forks", "open_issues", "contributors", "commits_30d",
                    "readme_length", "releases", "topics_count", "size_kb", "age_days", "days_since_push"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    bool_cols = ["has_ci", "has_tests", "has_license"]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    # Derived signals
    df["commits_per_day"] = df["commits_30d"] / 30.0
    df["forks_per_star"] = df.apply(
        lambda r: r["forks"] / r["stars"] if r["stars"] > 0 else 0.0, axis=1
    )
    df["issues_per_contributor"] = df.apply(
        lambda r: r["open_issues"] / r["contributors"] if r["contributors"] > 0 else r["open_issues"],
        axis=1,
    )
    df["repo_age_years"] = df["age_days"] / 365.0
    df["is_recently_active"] = df["days_since_push"] < 90
    df["has_releases"] = df["releases"] > 0
    df["engineering_score"] = (
        df["has_ci"].astype(int) * 3
        + df["has_tests"].astype(int) * 2
        + df["has_license"].astype(int)
        + df["has_releases"].astype(int) * 2
        + (df["contributors"] > 1).astype(int) * 2
        + (df["readme_length"] > 1000).astype(int)
    )

    df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)
    log.info(f"Processed dataset → {output_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    preprocess()
