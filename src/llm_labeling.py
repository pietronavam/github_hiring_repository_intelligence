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


def call_llm(summary: str, retries: int = 3, max_rate_limit_wait: int = 120) -> dict | None:
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
                if wait > max_rate_limit_wait:
                    log.warning(f"Rate limit wait {wait}s exceeds max {max_rate_limit_wait}s — using rule-based fallback")
                    return None
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


def rule_based_label(row: pd.Series) -> tuple[str, float, str]:
    """
    Deterministic fallback labeler based on the engineering_score composite
    and key signals. Used when LLM API is unavailable or rate-limited.

    This rule set operationalizes the hiring rubric explicitly:
      - low_value:  no activity, tiny, or fully abandoned
      - template:   high stars but zero engineering depth (viral boilerplate pattern)
      - intern:     solo + no CI/tests + minimal documentation
      - junior:     some structure, maybe CI, small team
      - senior:     CI + tests + contributors + releases
      - lead:       all signals strong, large team, many releases
    """
    score = int(row.get("engineering_score", 0))
    stars = int(row.get("stars", 0))
    contribs = int(row.get("contributors", 1))
    commits = int(row.get("commits_30d", 0))
    days_push = int(row.get("days_since_push", 9999))
    releases = int(row.get("releases", 0))
    size = int(row.get("size_kb", 0))
    has_ci = bool(row.get("has_ci", False))
    has_tests = bool(row.get("has_tests", False))

    if size < 5 or (days_push > 365 and stars < 5 and commits == 0):
        return "low_value", 0.75, "Very small or long-abandoned repo with no activity"

    if stars > 200 and contribs <= 2 and score <= 3 and releases == 0:
        return "template", 0.72, "High stars but no engineering depth — likely viral template"

    if score >= 9 and contribs > 10 and releases > 5:
        return "lead", 0.78, "Strong engineering signals across all dimensions with large team"

    if score >= 6 and (has_ci or has_tests) and (contribs > 2 or releases > 0):
        return "senior", 0.74, "Good CI/CD, tests, and collaboration signals"

    if score >= 3 and (has_ci or contribs > 1 or releases > 0):
        return "junior", 0.70, "Basic structure with some engineering practices"

    return "intern", 0.68, "Single contributor, minimal structure, limited practices"


def label_repositories(
    input_path: Path = SUMMARIZED_PATH,
    output_path: Path = LABELED_PATH,
    confidence_threshold: float = 0.6,
    max_rate_limit_wait: int = 120,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    log.info(f"Labeling {len(df)} repositories via LLM (fallback: rule-based)")

    labels, confidences, reasons, methods = [], [], [], []
    failed = 0

    for i, row in df.iterrows():
        result = call_llm(row["llm_summary"], max_rate_limit_wait=max_rate_limit_wait)
        if result and result.get("label") in CATEGORIES:
            labels.append(result["label"])
            confidences.append(float(result.get("confidence", 0.5)))
            reasons.append(result.get("reason", ""))
            methods.append("llm")
        else:
            # Fallback to rule-based label
            lbl, conf, reason = rule_based_label(row)
            labels.append(lbl)
            confidences.append(conf)
            reasons.append(reason)
            methods.append("rule_based")
            if result is None:
                failed += 1

        if (i + 1) % 20 == 0:
            llm_count = methods.count("llm")
            log.info(f"Progress: {i+1}/{len(df)} labeled — LLM:{llm_count}, rules:{i+1-llm_count}")
            # Save checkpoint
            df_temp = df.iloc[:i+1].copy()
            df_temp["label"] = labels
            df_temp["confidence"] = confidences
            df_temp["label_reason"] = reasons
            df_temp["label_method"] = methods
            df_temp.to_csv(output_path, index=False)
            time.sleep(0.5)
        else:
            time.sleep(0.3)

    df["label"] = labels
    df["confidence"] = confidences
    df["label_reason"] = reasons
    df["label_method"] = methods

    df.to_csv(output_path, index=False)
    llm_count = methods.count("llm")
    log.info(f"Done. LLM labels: {llm_count}, Rule-based: {len(df)-llm_count}")
    log.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    label_repositories()
