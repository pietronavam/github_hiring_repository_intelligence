"""
Full pipeline runner — executes all stages in order.
Usage: python run_pipeline.py [--skip-collect] [--skip-label] [--skip-train]
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run the full hiring intelligence pipeline")
    parser.add_argument("--skip-collect", action="store_true", help="Skip GitHub data collection")
    parser.add_argument("--skip-label", action="store_true", help="Skip LLM labeling")
    parser.add_argument("--skip-train", action="store_true", help="Skip BERT training")
    args = parser.parse_args()

    if not args.skip_collect:
        print("\n[1/6] Collecting GitHub repositories...")
        from src.github_collector import collect
        collect()

    print("\n[2/6] Preprocessing...")
    from src.preprocessing import preprocess
    preprocess()

    print("\n[3/6] Building text summaries...")
    from src.summarization import summarize
    summarize()

    if not args.skip_label:
        print("\n[4/6] LLM weak labeling (GitHub Models API)...")
        from src.llm_labeling import label_repositories
        label_repositories()

    if not args.skip_train:
        print("\n[5/6] Fine-tuning DistilBERT...")
        from src.train import train
        train()

    print("\n[6a/6] Evaluation...")
    from src.evaluation import run_evaluation
    run_evaluation()

    print("\n[6b/6] Generating visualizations...")
    from src.visualization import generate_all
    generate_all()

    print("\nPipeline complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
