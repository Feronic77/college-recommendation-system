#!/usr/bin/env python3
"""
============================================================
  Project : Engineering College Recommendation System
  Name    : Nishal KV
  Roll No : 67
============================================================

main.py
CLI entry point for the Engineering College Recommendation System.

Usage
-----
Interactive mode:
    python main.py

Non-interactive mode (for automation / testing):
    python main.py --branch "Computer Science" --budget 5 --location 5
"""

import argparse
import os
import sys

import pandas as pd
from tabulate import tabulate

from generate_dataset import generate_synthetic_data, save_dataset
from recommendation_engine import (
    recommend_colleges,
    train_ml_model,
    predict_with_model,
    SCORE_FEATURES,
)
from visualizations import generate_all_charts

# ─── Constants ────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "college_dataset.csv"
)

BRANCHES = [
    "Computer Science",
    "Electronics & Communication",
    "Mechanical Engineering",
    "Civil Engineering",
    "Electrical Engineering",
    "Information Technology",
    "Chemical Engineering",
    "Biotechnology",
]

DISPLAY_COLS = [
    "College Name",
    "Branch Offered",
    "Placement Rate (%)",
    "Average Package (LPA)",
    "Accreditation",
    "Infrastructure Score (1-10)",
    "Faculty Rating (1-10)",
    "Tuition Fees (LPA)",
    "Location Score (1-10)",
    "Student Satisfaction Rating (1-10)",
    "Final Score",
    "Recommendation",
]


# ─── Helpers ──────────────────────────────────────────────────────────

def _banner():
    print("\n" + "=" * 65)
    print("   🎓  ENGINEERING COLLEGE RECOMMENDATION SYSTEM  🎓")
    print("=" * 65)


def _print_table(df: pd.DataFrame, max_rows: int = 15):
    """Pretty-print the top results as a table."""
    show = df.head(max_rows)
    cols = [c for c in DISPLAY_COLS if c in show.columns]
    print("\n" + tabulate(show[cols], headers="keys", tablefmt="fancy_grid",
                          showindex=True, floatfmt=".2f"))


def _print_summary(df: pd.DataFrame):
    """Print high-level statistics about the recommendations."""
    print("\n📈  Summary Statistics")
    print("-" * 40)
    counts = df["Recommendation"].value_counts()
    for cat in ["Highly Recommended", "Recommended",
                "Moderately Suitable", "Not Recommended"]:
        n = counts.get(cat, 0)
        print(f"  {cat:<25s}  {n:>3d} colleges")
    print(f"\n  Average Final Score : {df['Final Score'].mean():.2f}")
    print(f"  Highest Final Score: {df['Final Score'].max():.2f}")
    print(f"  Lowest  Final Score: {df['Final Score'].min():.2f}")


def _interactive_input():
    """Prompt the user for preferences interactively."""
    print("\nAvailable branches:")
    for i, b in enumerate(BRANCHES, 1):
        print(f"  {i}. {b}")

    while True:
        choice = input("\nEnter branch number or name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(BRANCHES):
            branch = BRANCHES[int(choice) - 1]
            break
        # Accept partial text match
        matches = [b for b in BRANCHES if choice.lower() in b.lower()]
        if matches:
            branch = matches[0]
            break
        print("  ⚠  Invalid choice, try again.")

    budget_str = input("Maximum tuition budget in LPA (press Enter to skip): ").strip()
    budget = float(budget_str) if budget_str else None

    loc_str = input("Minimum location score 1-10 (press Enter to skip): ").strip()
    loc = float(loc_str) if loc_str else None

    return branch, budget, loc


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Engineering College Recommendation System"
    )
    parser.add_argument("--branch", type=str, default=None,
                        help="Preferred branch of study")
    parser.add_argument("--budget", type=float, default=None,
                        help="Max tuition fee in LPA")
    parser.add_argument("--location", type=float, default=None,
                        help="Minimum location score (1-10)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration of the dataset")
    args = parser.parse_args()

    _banner()

    # 1. Load or generate dataset
    if os.path.exists(DATASET_PATH) and not args.regenerate:
        print("\n📂 Loading existing dataset…")
        df = pd.read_csv(DATASET_PATH)
        print(f"   {len(df)} records loaded from {DATASET_PATH}")
    else:
        print("\n🔧 Generating synthetic dataset…")
        df = generate_synthetic_data()
        save_dataset(df, DATASET_PATH)

    # 2. Train ML model on full data
    print("\n🤖 Training RandomForest classification model…")
    model, report = train_ml_model(df)
    print("   Classification Report (on 25 % hold-out):\n")
    print(report)

    # 3. Get user preferences
    if args.branch:
        branch, budget, loc = args.branch, args.budget, args.location
    else:
        branch, budget, loc = _interactive_input()

    print(f'\n🔍 Searching for "{branch}" colleges', end="")
    if budget:
        print(f" | budget ≤ {budget} LPA", end="")
    if loc:
        print(f" | location ≥ {loc}", end="")
    print(" …\n")

    # 4. Run recommendation engine
    results = recommend_colleges(df, branch, budget, loc)

    if results.empty:
        print("😔 No colleges matched your criteria. Try relaxing filters.")
        sys.exit(0)

    # 5. Also attach ML predictions for comparison
    results["ML Prediction"] = predict_with_model(model, results).values

    # 6. Display results
    _print_table(results)
    _print_summary(results)

    # 7. Generate visualisations
    print("\n🎨 Generating visualisations…")
    chart_paths = generate_all_charts(results, top_n=min(10, len(results)))
    print(f"\n✅ Done!  {len(chart_paths)} charts saved to the output/ folder.")


if __name__ == "__main__":
    main()
