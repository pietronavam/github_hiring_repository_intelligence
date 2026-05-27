"""
GitHub Repository Collector

Sampling strategy (analytical justification):
  We stratify by star count as a proxy for visibility and adoption. Stars correlate
  loosely with quality but also capture templates and viral repos, which is why we
  deliberately over-sample extremes. Four tiers give us diversity across the
  engineering-maturity spectrum without biasing the dataset toward popular projects.

  Tier 0: 0–2 stars  → low_value / intern candidates
  Tier 1: 3–50 stars → intern / junior candidates
  Tier 2: 51–500 stars → junior / senior candidates
  Tier 3: 501+ stars → senior / lead candidates
  Tier 4: is:template → template candidates

  Per repo we collect 12 signals:
    stars, forks, open_issues, contributors, commits_30d,
    has_ci, readme_length, has_tests, releases,
    age_days, days_since_push, topics_count

  Rationale: these signals cover code activity (commits), collaboration (contributors,
  PRs-via-issues), engineering practices (CI, tests), documentation (README), and
  project health (releases, recency).
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from src.utils import get_github_headers, handle_rate_limit, setup_logging, GITHUB_API_BASE

log = setup_logging("collector")

RAW_PATH = Path("data/raw/repositories.csv")
SAMPLE_PER_TIER = 70  # ~350 total repos


def search_repos(query: str, per_page: int = 30, pages: int = 3) -> list[dict]:
    results = []
    for page in range(1, pages + 1):
        url = f"{GITHUB_API_BASE}/search/repositories"
        params = {"q": query, "per_page": per_page, "page": page, "sort": "updated"}
        r = requests.get(url, headers=get_github_headers(), params=params)
        handle_rate_limit(r)
        if r.status_code != 200:
            log.warning(f"Search failed ({r.status_code}): {query}")
            break
        items = r.json().get("items", [])
        results.extend(items)
        if len(items) < per_page:
            break
        time.sleep(0.5)
    return results


def get_contributor_count(owner: str, repo: str) -> int:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contributors"
    r = requests.get(url, headers=get_github_headers(), params={"per_page": 1, "anon": "false"})
    handle_rate_limit(r)
    if r.status_code != 200:
        return 0
    # GitHub returns Link header with last page number = total pages ≈ contributor count
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        try:
            last_url = [p.strip().split(";")[0].strip("<>") for p in link.split(",") if 'rel="last"' in p][0]
            return int(last_url.split("page=")[-1])
        except Exception:
            pass
    return len(r.json())


def get_recent_commits(owner: str, repo: str, days: int = 30) -> int:
    since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Use stats/commit_activity endpoint (less API calls)
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/stats/commit_activity"
    r = requests.get(url, headers=get_github_headers())
    handle_rate_limit(r)
    if r.status_code != 200 or not r.json():
        return 0
    # Last ~4 weeks
    weeks = r.json()[-4:] if len(r.json()) >= 4 else r.json()
    return sum(w.get("total", 0) for w in weeks)


def has_ci_workflows(owner: str, repo: str) -> bool:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/.github/workflows"
    r = requests.get(url, headers=get_github_headers())
    handle_rate_limit(r)
    return r.status_code == 200


def get_readme_length(owner: str, repo: str) -> int:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    r = requests.get(url, headers=get_github_headers())
    handle_rate_limit(r)
    if r.status_code != 200:
        return 0
    return r.json().get("size", 0)


def has_test_directory(owner: str, repo: str) -> bool:
    for name in ["test", "tests", "spec", "__tests__"]:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{name}"
        r = requests.get(url, headers=get_github_headers())
        handle_rate_limit(r)
        if r.status_code == 200:
            return True
    return False


def get_release_count(owner: str, repo: str) -> int:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/releases"
    r = requests.get(url, headers=get_github_headers(), params={"per_page": 1})
    handle_rate_limit(r)
    if r.status_code != 200:
        return 0
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        try:
            last_url = [p.strip().split(";")[0].strip("<>") for p in link.split(",") if 'rel="last"' in p][0]
            return int(last_url.split("page=")[-1])
        except Exception:
            pass
    return len(r.json())


def extract_signals(item: dict) -> dict | None:
    owner = item["owner"]["login"]
    repo = item["name"]
    full_name = item["full_name"]

    try:
        created_at = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
        pushed_at = datetime.fromisoformat(item["pushed_at"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        age_days = (now - created_at).days
        days_since_push = (now - pushed_at).days

        log.info(f"Extracting signals for {full_name}")

        contributors = get_contributor_count(owner, repo)
        commits_30d = get_recent_commits(owner, repo)
        has_ci = has_ci_workflows(owner, repo)
        readme_len = get_readme_length(owner, repo)
        has_tests = has_test_directory(owner, repo)
        releases = get_release_count(owner, repo)

        return {
            "full_name": full_name,
            "owner": owner,
            "repo": repo,
            "description": item.get("description") or "",
            "language": item.get("language") or "unknown",
            "stars": item.get("stargazers_count", 0),
            "forks": item.get("forks_count", 0),
            "open_issues": item.get("open_issues_count", 0),
            "topics_count": len(item.get("topics", [])),
            "topics": ",".join(item.get("topics", [])),
            "size_kb": item.get("size", 0),
            "has_license": item.get("license") is not None,
            "is_fork": item.get("fork", False),
            "age_days": age_days,
            "days_since_push": days_since_push,
            "contributors": contributors,
            "commits_30d": commits_30d,
            "has_ci": has_ci,
            "readme_length": readme_len,
            "has_tests": has_tests,
            "releases": releases,
        }
    except Exception as e:
        log.warning(f"Failed extracting {full_name}: {e}")
        return None


def collect(output_path: Path = RAW_PATH) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tiers = [
        ("stars:0..2 fork:false pushed:>2022-01-01", SAMPLE_PER_TIER),
        ("stars:3..50 fork:false pushed:>2022-01-01", SAMPLE_PER_TIER),
        ("stars:51..500 pushed:>2022-01-01", SAMPLE_PER_TIER),
        ("stars:>500 pushed:>2022-01-01", SAMPLE_PER_TIER),
        ("is:template stars:>5", SAMPLE_PER_TIER),
    ]

    all_repos = []
    seen = set()

    for query, target in tiers:
        log.info(f"Searching tier: {query}")
        items = search_repos(query, per_page=30, pages=3)
        count = 0
        for item in items:
            if count >= target:
                break
            if item["full_name"] in seen:
                continue
            seen.add(item["full_name"])
            signals = extract_signals(item)
            if signals:
                all_repos.append(signals)
                count += 1
            time.sleep(0.3)

    df = pd.DataFrame(all_repos)
    df.to_csv(output_path, index=False)
    log.info(f"Collected {len(df)} repositories → {output_path}")
    return df


if __name__ == "__main__":
    collect()
