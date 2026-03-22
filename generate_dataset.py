"""
============================================================
  Project : Engineering College Recommendation System
  Name    : Nishal KV
  Roll No : 67
============================================================

generate_dataset.py
Generates a synthetic benchmark dataset of engineering colleges.
"""

import numpy as np
import pandas as pd
import os

# Seed for reproducibility
np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────
NUM_COLLEGES = 200

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

COLLEGE_PREFIXES = [
    "Indian Institute of Technology",
    "National Institute of Technology",
    "Birla Institute of Technology",
    "Vellore Institute of Technology",
    "SRM Institute of Science",
    "Manipal Institute of Technology",
    "International Institute of Technology",
    "Delhi Technological University",
    "Jadavpur University",
    "Anna University",
    "Amity University",
    "Lovely Professional University",
    "Chandigarh University",
    "Thapar Institute",
    "PSG College of Technology",
    "RV College of Engineering",
    "BMS College of Engineering",
    "MVSR Engineering College",
    "Osmania University",
    "JNTU College of Engineering",
    "BITS Pilani",
    "IIIT Hyderabad",
    "IIIT Delhi",
    "College of Engineering Pune",
    "Visvesvaraya Technological University",
]

CITY_SUFFIXES = [
    "Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Jaipur", "Lucknow", "Ahmedabad",
    "Bhopal", "Chandigarh", "Nagpur", "Coimbatore", "Indore",
    "Trichy", "Warangal", "Surathkal", "Roorkee", "Kharagpur",
]


def generate_synthetic_data(num_colleges: int = NUM_COLLEGES) -> pd.DataFrame:
    """
    Create a synthetic dataset of engineering colleges with realistic
    distributions for each attribute.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per college-branch combination (may exceed
        `num_colleges` rows because each college offers 1-4 branches).
    """
    records = []
    college_id = 0

    while college_id < num_colleges:
        # Pick a random college name
        prefix = np.random.choice(COLLEGE_PREFIXES)
        city = np.random.choice(CITY_SUFFIXES)
        college_name = f"{prefix}, {city}"

        # Each college offers 1-4 branches
        num_branches = np.random.randint(1, 5)
        offered_branches = np.random.choice(BRANCHES, size=num_branches, replace=False)

        # College-level attributes (shared across branches)
        accreditation = np.random.choice(["Yes", "No"], p=[0.7, 0.3])
        infrastructure = round(np.clip(np.random.normal(6.5, 1.8), 1, 10), 1)
        location_score = round(np.clip(np.random.normal(6.0, 2.0), 1, 10), 1)
        base_tuition = round(np.clip(np.random.lognormal(mean=1.0, sigma=0.5), 0.5, 25), 2)

        for branch in offered_branches:
            # Branch-level attributes (vary per branch)
            placement_rate = round(np.clip(np.random.normal(65, 18), 10, 100), 1)
            avg_package = round(np.clip(np.random.lognormal(mean=1.2, sigma=0.55), 1.5, 45), 2)
            faculty_rating = round(np.clip(np.random.normal(6.0, 1.5), 1, 10), 1)
            student_satisfaction = round(np.clip(np.random.normal(6.5, 1.5), 1, 10), 1)

            # Slight tuition variation per branch
            tuition = round(base_tuition * np.random.uniform(0.9, 1.1), 2)

            records.append({
                "College Name": college_name,
                "Branch Offered": branch,
                "Placement Rate (%)": placement_rate,
                "Average Package (LPA)": avg_package,
                "Accreditation": accreditation,
                "Infrastructure Score (1-10)": infrastructure,
                "Faculty Rating (1-10)": faculty_rating,
                "Tuition Fees (LPA)": tuition,
                "Location Score (1-10)": location_score,
                "Student Satisfaction Rating (1-10)": student_satisfaction,
            })

            college_id += 1
            if college_id >= num_colleges:
                break

    df = pd.DataFrame(records)
    return df


def save_dataset(df: pd.DataFrame, path: str = "college_dataset.csv") -> str:
    """Save the dataset to a CSV file and return the absolute path."""
    abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    df.to_csv(abs_path, index=False)
    print(f"✅ Dataset saved to {abs_path}  ({len(df)} records)")
    return abs_path


# ─── Standalone usage ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_synthetic_data()
    save_dataset(df)
    print(df.head(10).to_string(index=False))
