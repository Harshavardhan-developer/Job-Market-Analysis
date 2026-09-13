"""
prediction.py
--------------
Loads the trained best model + metadata and exposes a simple function to
predict salary for a new job description, along with a data-driven salary
range and human-readable "important factors" explanation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return model, metadata


def build_input_row(
    job_title, years_experience, experience_level, location,
    education_level, industry, company_size, employment_type,
    remote_ratio, skills, metadata,
):
    """Builds a single-row DataFrame matching the training feature schema."""
    top_skills = metadata["top_skills"]
    skill_cols = metadata["skill_cols"]

    row = {
        "Years of Experience": years_experience,
        "Remote Ratio": remote_ratio,
        "Skill Count": len(skills),
        "Job Title": job_title,
        "Location": location,
        "Experience Level": experience_level,
        "Employment Type": employment_type,
        "Industry": industry,
        "Company Size": company_size,
        "Education Level": education_level,
    }
    for skill in top_skills:
        col = f"skill_{skill.lower().replace(' ', '_')}"
        row[col] = 1 if skill in skills else 0

    feature_cols = metadata["feature_cols"]
    return pd.DataFrame([row])[feature_cols]


def predict_salary(
    job_title, years_experience, experience_level, location,
    education_level, industry, company_size, employment_type,
    remote_ratio, skills,
):
    model, metadata = load_artifacts()
    X = build_input_row(
        job_title, years_experience, experience_level, location,
        education_level, industry, company_size, employment_type,
        remote_ratio, skills, metadata,
    )
    predicted = float(model.predict(X)[0])

    # Data-driven salary range: use the model's own test-set error (RMSE)
    # of the best model to build a realistic +/- band around the point estimate,
    # rather than an arbitrary fixed percentage.
    best_model_name = metadata["best_model"]
    rmse = metadata["results"][best_model_name]["RMSE"]
    low = max(0, predicted - rmse)
    high = predicted + rmse

    # Important factors: derived from the actual trained feature importance list.
    factors = []
    for feat, importance in metadata["feature_importance"][:6]:
        label = _humanize_feature(feat)
        factors.append({"factor": label, "importance": round(importance, 4)})

    return {
        "predicted_salary": round(predicted, 2),
        "salary_range": (round(low, 2), round(high, 2)),
        "model_used": best_model_name,
        "important_factors": factors,
    }


def _humanize_feature(feat_name: str) -> str:
    if feat_name.startswith("skill_"):
        return "Skill: " + feat_name.replace("skill_", "").replace("_", " ").title()
    for prefix in ["Job Title_", "Location_", "Experience Level_", "Employment Type_",
                   "Industry_", "Company Size_", "Education Level_"]:
        if feat_name.startswith(prefix):
            base = prefix.rstrip("_")
            value = feat_name[len(prefix):]
            return f"{base}: {value}"
    return feat_name


if __name__ == "__main__":
    result = predict_salary(
        job_title="Data Scientist",
        years_experience=4,
        experience_level="Mid",
        location="Berlin, Germany",
        education_level="Master",
        industry="Technology",
        company_size="Medium",
        employment_type="Full-time",
        remote_ratio=50,
        skills=["Python", "SQL", "Machine Learning"],
    )
    print(json.dumps(result, indent=2))
