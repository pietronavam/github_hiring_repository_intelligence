"""
LLM Weak Labeling (Stage 3)

Why weak supervision instead of manual labeling?
  Manually labeling 300+ repositories by seniority level would require a
  senior engineer spending ~2 minutes per repo = 10+ hours. LLMs have been
  trained on GitHub data, code reviews, and engineering discussions, giving
  them a reasonable prior on what constitutes "intern vs senior" code.
  The labels are noisy (weak) but consistent enough to train a classifier.

Prompt design choices:
  1. We give the LLM the rubric explicitly so labels are consistent across calls.
  2. We include the composite engineering_score as a signal anchor to reduce
     variance in borderline cases.
  3. We ask for JSON output with a confidence field so we can filter low-
     confidence labels before training (label quality threshold).
  4. We use few-shot examples in the system prompt to calibrate output format.

API: GitHub Models (gpt-4o-mini) — free with GitHub PAT, OpenAI-compatible.
Fallback: any OpenAI-compatible endpoint via LLM_BASE_URL env variable.
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from src.utils import (
    setup_logging, GITHUB_MODELS_BASE, GITHUB_TOKEN,
    CATEGORIES, CATEGORY_DESCRIPTIONS
)

log = setup_logging("llm_labeling")

SUMMARIZED_PATH = Path("data/processed/repositories_summarized.csv")
LABELED_PATH = Path("data/labeled/repositories_labeled.csv")

SYSTEM_PROMPT = """You are an expert software engineering recruiter with 15 years of experience.
Your task is to classify GitHub repositories by the engineering maturity level they reflect.

Categories and their definitions:
- intern: Simple scripts/notebooks, single contributor, no tests, no CI/CD, minimal structure
- junior: Basic project structure, limited tests, basic CI or none, 1-3 contributors
- senior: Well-structured project with CI/CD, test coverage, good docs, multiple contributors, releases
- lead: Complex system/library, many contributors, advanced architecture, regular releases, production-grade
- template: Boilerplate, starter template, or clone with no original engineering work
- low_value: Abandoned, empty, trivial, or test repository with no real content

Rules:
- Return ONLY valid JSON, no extra text
- Be strict: a repo with 1000 stars can still be a template
- Focus on engineering practices, not popularity
- The engineering_score field (0-11) is a composite signal to help you calibrate

Output format (JSON only):
{"label": "<one of the 6 categories>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}

Examples:
Input: stars=2, contributors=1, has_ci=false, has_tests=false, commits_30d=0, score=0, age=3y, inactive
Output: {"label": "low_value", "confidence": 0.95, "reason": "Abandoned single-contributor repo with no engineering practices"}

Input: stars=850, contributors=45, has_ci=true, has_tests=true, commits_30d=120, score=11, releases=28, active
Output: {"label": "lead", "confidence": 0.9, "reason": "Production-grade project with strong team, CI, tests, and frequent releases"}

Input: stars=5000, contributors=1, has_ci=false, has_tests=false, commits_30d=0, score=1, age=2y, inactive
Output: {"label": "template", "confidence": 0.85, "reason": "High stars but no engineering depth — typical of boilerplate or viral template"}"""


def call_llm(summary: str, retries: int = 3) -> dict | None:
    base_url = os.getenv("LLM_BASE_URL", GITHUB_MODELS_BASE)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    token = os.getenv("LLM_TOKEN", GITHUB_TOKEN)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this repository:\n\n{summary}"},
        ],
        "temperature": 0.1,
        "max_tokens": 150,
    }

    for attempt in range(retries):
        try:
            r = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10)) + 2
                log.warning(f"Rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                log.warning(f"LLM call failed ({r.status_code}): {r.text[:200]}")
                return None

            content = r.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error on attempt {attempt+1}: {e}")
        except Exception as e:
            log.warning(f"LLM call error on attempt {attempt+1}: {e}")
            time.sleep(2)

    return None


def label_repositories(
    input_path: Path = SUMMARIZED_PATH,
    output_path: Path = LABELED_PATH,
    confidence_threshold: float = 0.6,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    log.info(f"Labeling {len(df)} repositories via LLM")

    labels, confidences, reasons = [], [], []
    failed = 0

    for i, row in df.iterrows():
        result = call_llm(row["llm_summary"])
        if result and result.get("label") in CATEGORIES:
            labels.append(result["label"])
            confidences.append(float(result.get("confidence", 0.5)))
            reasons.append(result.get("reason", ""))
        else:
            labels.append(None)
            confidences.append(0.0)
            reasons.append("labeling_failed")
            failed += 1

        if (i + 1) % 20 == 0:
            log.info(f"Progress: {i+1}/{len(df)} repos labeled ({failed} failed)")
            time.sleep(1)  # brief pause every 20 calls
        else:
            time.sleep(0.3)

    df["label"] = labels
    df["confidence"] = confidences
    df["label_reason"] = reasons

    # Keep only high-confidence labels for training
    df_filtered = df[df["confidence"] >= confidence_threshold].dropna(subset=["label"]).copy()

    df.to_csv(output_path, index=False)
    log.info(f"All labels saved → {output_path}")
    log.info(f"High-confidence labels: {len(df_filtered)}/{len(df)} (threshold={confidence_threshold})")
    log.info(f"Label distribution:\n{df_filtered['label'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    label_repositories()
