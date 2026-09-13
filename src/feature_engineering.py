"""
feature_engineering.py
-----------------------
Builds the feature matrix used for ML training and prediction, and exposes
the exact list of numeric / categorical / skill columns used by the
preprocessing pipeline so training and inference stay consistent.
"""

import pandas as pd

NUMERIC_FEATURES = ["Years of Experience", "Remote Ratio", "Skill Count"]

CATEGORICAL_FEATURES = [
    "Job Title", "Location", "Experience Level", "Employment Type",
    "Industry", "Company Size", "Education Level",
]

TARGET = "Salary_USD"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# A curated list of individual skill flags is also engineered (top-N most
# frequent skills become their own binary columns) to let tree models pick
# up on specific skill effects beyond raw skill count.
TOP_N_SKILLS = 15


def get_top_skills(df: pd.DataFrame, top_n: int = TOP_N_SKILLS):
    all_skills = df["Skills List"].explode()
    counts = all_skills.value_counts()
    return list(counts.head(top_n).index)


def add_skill_flags(df: pd.DataFrame, top_skills):
    df = df.copy()
    for skill in top_skills:
        col_name = f"skill_{skill.lower().replace(' ', '_')}"
        df[col_name] = df["Skills List"].apply(lambda lst: 1 if skill in lst else 0)
    return df


def build_feature_frame(df: pd.DataFrame, top_skills):
    """Return the feature matrix (X) and target vector (y) plus the final
    feature column list (including skill flags) for model training/inference."""
    df = add_skill_flags(df, top_skills)
    skill_cols = [f"skill_{s.lower().replace(' ', '_')}" for s in top_skills]
    feature_cols = ALL_FEATURES + skill_cols

    X = df[feature_cols]
    y = df[TARGET] if TARGET in df.columns else None
    return X, y, feature_cols, skill_cols
