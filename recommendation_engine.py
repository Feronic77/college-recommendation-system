"""
============================================================
  Project : Engineering College Recommendation System
  Name    : Nishal KV
  Roll No : 67
============================================================

recommendation_engine.py
Core recommendation logic: filtering, weighted scoring, labelling,
and an optional RandomForest ML model for generalisation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import textwrap


# ─── Feature weights ──────────────────────────────────────────────────
FEATURE_WEIGHTS = {
    "Placement Rate (%)":             0.25,
    "Average Package (LPA)":          0.20,
    "Faculty Rating (1-10)":          0.20,
    "Infrastructure Score (1-10)":    0.15,
    "Student Satisfaction Rating (1-10)": 0.10,
    "Location Score (1-10)":          0.10,
}

SCORE_FEATURES = list(FEATURE_WEIGHTS.keys())


# ─── Helpers ──────────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise the scoring features in [0, 1]."""
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[SCORE_FEATURES] = scaler.fit_transform(df[SCORE_FEATURES])
    return df_norm


def _compute_weighted_score(df_norm: pd.DataFrame) -> pd.Series:
    """Return the weighted composite score as a Series."""
    score = pd.Series(0.0, index=df_norm.index)
    for feat, w in FEATURE_WEIGHTS.items():
        score += df_norm[feat] * w
    return round(score * 100, 2)          # scale to 0-100


def _assign_labels(scores: pd.Series) -> pd.Series:
    """Map composite scores to recommendation categories."""
    p80 = scores.quantile(0.80)
    p60 = scores.quantile(0.60)
    p40 = scores.quantile(0.40)

    def _label(s):
        if s >= p80:
            return "Highly Recommended"
        elif s >= p60:
            return "Recommended"
        elif s >= p40:
            return "Moderately Suitable"
        else:
            return "Not Recommended"

    return scores.apply(_label)


# ─── Public API ───────────────────────────────────────────────────────

def recommend_colleges(
    df: pd.DataFrame,
    preferred_branch: str,
    budget_lpa: float = None,
    min_location_score: float = None,
) -> pd.DataFrame:
    """
    Filter, score, and label colleges.

    Parameters
    ----------
    df : pd.DataFrame
        Full college dataset.
    preferred_branch : str
        Branch the student wants to study.
    budget_lpa : float, optional
        Maximum tuition the student can afford (LPA).
    min_location_score : float, optional
        Minimum acceptable location score (1-10).

    Returns
    -------
    pd.DataFrame
        Filtered and scored dataframe sorted by Final Score (desc),
        with a "Recommendation" column.
    """
    # 1. Filter by branch (case-insensitive partial match)
    mask = df["Branch Offered"].str.lower().str.contains(
        preferred_branch.lower(), na=False
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return filtered

    # 2. Filter by budget
    if budget_lpa is not None:
        filtered = filtered[filtered["Tuition Fees (LPA)"] <= budget_lpa]

    # 3. Filter by min location score
    if min_location_score is not None:
        filtered = filtered[filtered["Location Score (1-10)"] >= min_location_score]

    if filtered.empty:
        return filtered

    # 4. Normalise & score
    norm = _normalise(filtered)
    filtered["Final Score"] = _compute_weighted_score(norm).values

    # 5. Assign labels
    filtered["Recommendation"] = _assign_labels(filtered["Final Score"])

    # 6. Sort
    filtered = filtered.sort_values("Final Score", ascending=False).reset_index(drop=True)
    filtered.index += 1          # 1-based rank
    filtered.index.name = "Rank"

    return filtered


# ─── ML Model ────────────────────────────────────────────────────────

def train_ml_model(df: pd.DataFrame):
    """
    Train a RandomForestClassifier on the full labelled dataset.

    Returns
    -------
    model : RandomForestClassifier
    report : str
        Classification report string.
    """
    # First, score the entire dataset so we have labels
    norm = _normalise(df)
    df = df.copy()
    df["Final Score"] = _compute_weighted_score(norm).values
    df["Recommendation"] = _assign_labels(df["Final Score"])

    X = df[SCORE_FEATURES].values
    y = df["Recommendation"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    return model, report


def predict_with_model(model, df_filtered: pd.DataFrame) -> pd.Series:
    """Use the trained model to predict labels for new/filtered data."""
    X = df_filtered[SCORE_FEATURES].values
    return pd.Series(model.predict(X), index=df_filtered.index, name="ML Prediction")


# ─── Generative Component ─────────────────────────────────────────────

class GenerativeReport:
    """
    Simulates a Generative AI module that creates personalized advice 
    for students based on their top college recommendations.
    """
    
    @staticmethod
    def generate_summary(student_name, top_colleges: pd.DataFrame) -> str:
        """Generate a narrative summary for the top 3 recommendations."""
        if top_colleges.empty:
            return "No colleges found matching your criteria."
            
        summary = f"\n📝  PERSONALIZED ADVICE FOR {student_name.upper()}\n"
        summary += "=" * 50 + "\n"
        
        for i, (idx, row) in enumerate(top_colleges.head(3).iterrows(), 1):
            college = row['College Name']
            branch = row['Branch Offered']
            score = row['Final Score']
            placement = row['Placement Rate (%)']
            lpa = row['Average Package (LPA)']
            
            blurb = (
                f"Rank {i}: {college} ({branch}). With an overall score of {score:.1f}, "
                f"this institution is a strong contender. We highlight its {placement}% placement rate "
                f"and an impressive average package of {lpa} LPA. Based on its Faculty Rating of "
                f"{row['Faculty Rating (1-10)']}/10, you can expect high-quality mentorship."
            )
            summary += "\n" + "\n".join(textwrap.wrap(blurb, width=70)) + "\n"
            
        summary += "\n" + "=" * 50
        summary += "\nFinal Tip: Focus on colleges with high 'Student Satisfaction' titles."
        
        return summary
