"""
============================================================
  Project : Engineering College Recommendation System
  Name    : Nishal KV
  Roll No : 67
============================================================

visualizations.py
Generate informative charts for the college recommendation results.
Uses Matplotlib (Agg backend for headless environments) and Seaborn.
"""

import os
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ─── Style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {
    "Highly Recommended":  "#2ecc71",
    "Recommended":         "#3498db",
    "Moderately Suitable": "#f39c12",
    "Not Recommended":     "#e74c3c",
}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Chart 1: Top colleges bar chart ─────────────────────────────────

def plot_top_colleges(df: pd.DataFrame, top_n: int = 10):
    """Horizontal bar chart of the top-N recommended colleges by Final Score."""
    _ensure_output_dir()
    top = df.head(top_n).copy()
    top = top.iloc[::-1]  # reverse for bottom-up plotting

    colors = [PALETTE.get(r, "#95a5a6") for r in top["Recommendation"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(
        top["College Name"] + "\n(" + top["Branch Offered"] + ")",
        top["Final Score"],
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Final Score (0–100)", fontsize=12)
    ax.set_title(f"Top {top_n} Recommended Colleges", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 105)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in PALETTE.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "top_colleges_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved: {path}")
    return path


# ─── Chart 2: Category distribution pie chart ────────────────────────

def plot_category_distribution(df: pd.DataFrame):
    """Pie chart showing the share of each recommendation category."""
    _ensure_output_dir()
    counts = df["Recommendation"].value_counts()

    colors = [PALETTE.get(cat, "#95a5a6") for cat in counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.8,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for txt in autotexts:
        txt.set_fontsize(11)
        txt.set_fontweight("bold")

    ax.set_title("Recommendation Category Distribution",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "category_pie.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved: {path}")
    return path


# ─── Chart 3: Feature correlation heatmap ────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap of Pearson correlations among numeric features."""
    _ensure_output_dir()
    numeric_cols = [
        "Placement Rate (%)", "Average Package (LPA)",
        "Infrastructure Score (1-10)", "Faculty Rating (1-10)",
        "Tuition Fees (LPA)", "Location Score (1-10)",
        "Student Satisfaction Rating (1-10)",
    ]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        linewidths=0.5, square=True, ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title("Feature Correlation Heatmap",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved: {path}")
    return path


# ─── Chart 4: Placement rate box plot by category ────────────────────

def plot_placement_boxplot(df: pd.DataFrame):
    """Box plot comparing placement rates across recommendation labels."""
    _ensure_output_dir()

    order = ["Highly Recommended", "Recommended",
             "Moderately Suitable", "Not Recommended"]
    existing = [o for o in order if o in df["Recommendation"].unique()]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, x="Recommendation", y="Placement Rate (%)",
        order=existing,
        hue="Recommendation",
        hue_order=existing,
        palette=PALETTE,
        ax=ax,
        linewidth=1.2,
        legend=False,
    )
    ax.set_title("Placement Rate by Recommendation Category",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Placement Rate (%)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "placement_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved: {path}")
    return path


# ─── Generate all charts ─────────────────────────────────────────────

def generate_all_charts(df: pd.DataFrame, top_n: int = 10):
    """Convenience function: produce every visualisation."""
    paths = []
    paths.append(plot_top_colleges(df, top_n=top_n))
    paths.append(plot_category_distribution(df))
    paths.append(plot_correlation_heatmap(df))
    paths.append(plot_placement_boxplot(df))
    return paths
