"""
Streamlit dashboard for the Job Market Analysis & Salary Prediction project.

Run with:
    streamlit run dashboard/app.py
(run from the project root so the relative paths below resolve correctly)
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from visualization import (
    salary_distribution_chart, salary_by_category_chart, count_by_category_chart,
    remote_vs_onsite_chart, top_skills_chart, salary_vs_experience_chart,
    actual_vs_predicted_chart, feature_importance_chart,
)
from prediction import predict_salary

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "jobs_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

st.set_page_config(page_title="Job Market & Salary Analysis", layout="wide", page_icon="📊")


def _bootstrap_if_needed():
    """On a fresh deployment (e.g. Streamlit Community Cloud), the generated
    data/ and models/ directories may not exist in the repo (they're
    git-ignored to keep the repo lightweight). If they're missing, generate
    the dataset and train the models automatically on first load."""
    data_missing = not os.path.exists(DATA_PATH)
    models_missing = not os.path.exists(METADATA_PATH)

    if data_missing or models_missing:
        with st.spinner("First-time setup: generating dataset and training models (~30-60s)..."):
            from data_processing import run_pipeline
            if data_missing:
                run_pipeline(n_rows=6000, force_regenerate=True)
            from model_training import train_and_evaluate
            if models_missing or data_missing:
                train_and_evaluate()


_bootstrap_if_needed()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Skills List"] = df["Skills List"].apply(eval)
    return df


@st.cache_data
def load_metadata():
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        return json.load(f)


df = load_data()
metadata = load_metadata()

st.sidebar.title("📊 Job Market Analysis")
st.sidebar.caption("AI/ML Job Market Analysis & Salary Prediction System")
st.sidebar.info(
    "This project uses a **synthetic dataset** generated programmatically to "
    "simulate realistic job-market patterns, since a verified public dataset "
    "was not bundled. All metrics/charts are computed from this data."
)
page = st.sidebar.radio(
    "Navigate",
    [
        "1. Overview",
        "2. Job Market Analytics",
        "3. Salary Analytics",
        "4. Salary Prediction",
        "5. Skills Analysis",
        "6. ML Model Performance",
        "7. Job Role Explorer",
    ],
)

# --------------------------------------------------------------------------
# PAGE 1 — OVERVIEW
# --------------------------------------------------------------------------
if page.startswith("1"):
    st.title("Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    top_role = df["Job Title"].value_counts().idxmax()
    top_skill = df["Skills List"].explode().value_counts().idxmax()

    col1.metric("Total Jobs", f"{len(df):,}")
    col2.metric("Average Salary", f"${df['Salary_USD'].mean():,.0f}")
    col3.metric("Median Salary", f"${df['Salary_USD'].median():,.0f}")
    col4.metric("Most Common Role", top_role)
    col5.metric("Most Demanded Skill", top_skill)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(salary_distribution_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(count_by_category_chart(df, "Job Title"), use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 2 — JOB MARKET ANALYTICS
# --------------------------------------------------------------------------
elif page.startswith("2"):
    st.title("Job Market Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(count_by_category_chart(df, "Location"), use_container_width=True)
    with c2:
        st.plotly_chart(count_by_category_chart(df, "Experience Level"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(count_by_category_chart(df, "Employment Type"), use_container_width=True)
    with c4:
        st.plotly_chart(remote_vs_onsite_chart(df), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(count_by_category_chart(df, "Job Title"), use_container_width=True)
    with c6:
        st.plotly_chart(top_skills_chart(df, top_n=10), use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 3 — SALARY ANALYTICS
# --------------------------------------------------------------------------
elif page.startswith("3"):
    st.title("Salary Analytics")
    st.plotly_chart(salary_distribution_chart(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(salary_vs_experience_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(salary_by_category_chart(df, "Job Title"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(salary_by_category_chart(df, "Location"), use_container_width=True)
    with c4:
        st.plotly_chart(salary_by_category_chart(df, "Industry"), use_container_width=True)

    st.plotly_chart(salary_by_category_chart(df, "Company Size", top_n=3), use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 4 — SALARY PREDICTION
# --------------------------------------------------------------------------
elif page.startswith("4"):
    st.title("Salary Prediction")
    st.caption(f"Model in use: **{metadata['best_model']}** "
               f"(R² = {metadata['results'][metadata['best_model']]['R2']})")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            job_title = st.selectbox("Job Title", sorted(df["Job Title"].unique()))
            experience_level = st.selectbox("Experience Level", ["Entry", "Mid", "Senior", "Lead"])
            years_experience = st.slider("Years of Experience", 0, 20, 3)
        with c2:
            location = st.selectbox("Location", sorted(df["Location"].unique()))
            education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
            employment_type = st.selectbox("Employment Type", sorted(df["Employment Type"].unique()))
        with c3:
            industry = st.selectbox("Industry", sorted(df["Industry"].unique()))
            company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
            remote_ratio = st.select_slider("Remote Ratio (%)", options=[0, 50, 100], value=50)

        skills = st.multiselect("Skills", metadata["top_skills"], default=metadata["top_skills"][:3])
        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        result = predict_salary(
            job_title, years_experience, experience_level, location,
            education_level, industry, company_size, employment_type,
            remote_ratio, skills,
        )
        st.success(f"### Predicted Salary: ${result['predicted_salary']:,.0f} / year")
        low, high = result["salary_range"]
        st.write(f"**Estimated Salary Range:** ${low:,.0f} – ${high:,.0f}")
        st.write(f"**Model Used:** {result['model_used']}")

        st.write("**Important Contributing Factors:**")
        factor_df = pd.DataFrame(result["important_factors"])
        st.plotly_chart(
            px.bar(factor_df, x="importance", y="factor", orientation="h",
                   title="Top Factors Influencing This Prediction"),
            use_container_width=True,
        )

# --------------------------------------------------------------------------
# PAGE 5 — SKILLS ANALYSIS
# --------------------------------------------------------------------------
elif page.startswith("5"):
    st.title("Skills Analysis")
    st.plotly_chart(top_skills_chart(df, top_n=20), use_container_width=True)

    st.subheader("Skills by Job Role")
    role = st.selectbox("Select a Job Role", sorted(df["Job Title"].unique()))
    role_skills = df[df["Job Title"] == role]["Skills List"].explode().value_counts().head(10)
    st.plotly_chart(
        px.bar(role_skills, x=role_skills.values, y=role_skills.index, orientation="h",
               title=f"Top Skills for {role}", labels={"x": "Count", "y": "Skill"}),
        use_container_width=True,
    )

    st.subheader("Skill Demand by Location")
    location_sel = st.selectbox("Select a Location", sorted(df["Location"].unique()))
    loc_skills = df[df["Location"] == location_sel]["Skills List"].explode().value_counts().head(10)
    st.plotly_chart(
        px.bar(loc_skills, x=loc_skills.values, y=loc_skills.index, orientation="h",
               title=f"Top Skills in {location_sel}", labels={"x": "Count", "y": "Skill"}),
        use_container_width=True,
    )

    st.subheader("Skill → Salary Relationship")
    top_skills_list = df["Skills List"].explode().value_counts().head(12).index.tolist()
    skill_salary = {}
    for skill in top_skills_list:
        mask = df["Skills List"].apply(lambda lst: skill in lst)
        skill_salary[skill] = df.loc[mask, "Salary_USD"].mean()
    skill_salary_series = pd.Series(skill_salary).sort_values(ascending=False)
    st.plotly_chart(
        px.bar(skill_salary_series, x=skill_salary_series.index, y=skill_salary_series.values,
               title="Average Salary Associated with Each Skill",
               labels={"y": "Average Salary (USD)", "x": "Skill"}),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# PAGE 6 — ML MODEL PERFORMANCE
# --------------------------------------------------------------------------
elif page.startswith("6"):
    st.title("ML Model Performance")

    results_df = pd.DataFrame(metadata["results"]).T.reset_index().rename(columns={"index": "Model"})
    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(px.bar(results_df, x="Model", y="MAE", title="MAE by Model"), use_container_width=True)
    c2.plotly_chart(px.bar(results_df, x="Model", y="RMSE", title="RMSE by Model"), use_container_width=True)
    c3.plotly_chart(px.bar(results_df, x="Model", y="R2", title="R² by Model"), use_container_width=True)

    st.subheader(f"Actual vs Predicted — Best Model ({metadata['best_model']})")
    y_test = metadata["y_test"]
    y_pred = metadata["predictions_by_model"][metadata["best_model"]]
    st.plotly_chart(actual_vs_predicted_chart(y_test, y_pred, metadata["best_model"]), use_container_width=True)

    st.subheader("Feature Importance")
    st.plotly_chart(feature_importance_chart(metadata["feature_importance"]), use_container_width=True)

    st.subheader("Interpretation")
    top_factors = metadata["feature_importance"][:5]
    for feat, imp in top_factors:
        level = "Strong" if imp > 0.1 else ("Moderate" if imp > 0.03 else "Lower")
        st.write(f"- **{feat}** → {level} influence (importance = {imp:.4f})")

# --------------------------------------------------------------------------
# PAGE 7 — JOB ROLE EXPLORER (Advanced Feature)
# --------------------------------------------------------------------------
elif page.startswith("7"):
    st.title("Job Role Explorer")
    role = st.selectbox("Choose a Job Role", sorted(df["Job Title"].unique()))
    role_df = df[df["Job Title"] == role]

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Salary", f"${role_df['Salary_USD'].mean():,.0f}")
    c2.metric("Median Salary", f"${role_df['Salary_USD'].median():,.0f}")
    c3.metric("Number of Postings", f"{len(role_df):,}")

    low, high = role_df["Salary_USD"].quantile([0.1, 0.9])
    st.write(f"**Typical Salary Range (10th–90th percentile):** ${low:,.0f} – ${high:,.0f}")

    c4, c5 = st.columns(2)
    with c4:
        st.plotly_chart(top_skills_chart(role_df, top_n=10), use_container_width=True)
    with c5:
        exp_dist = role_df["Experience Level"].value_counts().reindex(["Entry", "Mid", "Senior", "Lead"]).fillna(0)
        st.plotly_chart(
            px.bar(exp_dist, x=exp_dist.index, y=exp_dist.values, title="Experience Distribution",
                   labels={"y": "Count", "x": "Experience Level"}),
            use_container_width=True,
        )

    c6, c7 = st.columns(2)
    with c6:
        top_locations = role_df["Location"].value_counts().head(8)
        st.plotly_chart(
            px.bar(top_locations, x=top_locations.index, y=top_locations.values,
                   title="Popular Locations", labels={"y": "Count", "x": "Location"}),
            use_container_width=True,
        )
    with c7:
        remote_pct = (role_df["Remote Ratio"] == 100).mean() * 100
        st.metric("Remote-Job Percentage", f"{remote_pct:.1f}%")
        top_companies = role_df["Company"].value_counts().head(8)
        st.plotly_chart(
            px.bar(top_companies, x=top_companies.index, y=top_companies.values,
                   title="Top Companies Hiring for This Role", labels={"y": "Count", "x": "Company"}),
            use_container_width=True,
        )
