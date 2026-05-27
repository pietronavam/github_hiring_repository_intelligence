import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API_BASE = "https://api.github.com"
GITHUB_MODELS_BASE = "https://models.inference.ai.azure.com"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")

CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]

CATEGORY_DESCRIPTIONS = {
    "intern":    "Simple script or notebook, single contributor, no tests, no CI/CD, minimal structure",
    "junior":    "Basic project with some structure, limited tests, basic CI or none, 1-3 contributors",
    "senior":    "Well-structured project with CI/CD, test coverage, good documentation, multiple contributors, releases",
    "lead":      "Complex system or library, many contributors, advanced architecture, regular releases, production-grade quality",
    "template":  "Boilerplate, starter template, or clone with no original work",
    "low_value": "Abandoned, empty, trivial, or test repository with no real content",
}


def get_github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name)


def handle_rate_limit(response) -> None:
    remaining = int(response.headers.get("X-RateLimit-Remaining", 100))
    if remaining < 15:
        reset_ts = int(response.headers.get("X-RateLimit-Reset", time.time() + 61))
        wait = max(reset_ts - time.time() + 3, 0)
        logging.getLogger("utils").warning(
            f"Rate limit low ({remaining} left). Waiting {wait:.0f}s"
        )
        time.sleep(wait)
