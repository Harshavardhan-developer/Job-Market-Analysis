"""
visualization.py
-----------------
Reusable Plotly chart-builder functions shared between the Streamlit
dashboard and the EDA notebook, so charts stay consistent across both.
"""

import pandas as pd
import plotly.express as px


def salary_distribution_chart(df: pd.DataFrame):
    fig = px.histogram(df, x="Salary_USD", nbins=50, title="Salary Distribution (USD/year)")
    fig.update_layout(xaxis_title="Salary (USD)", yaxis_title="Count")
    return fig


def salary_by_category_chart(df: pd.DataFrame, category: str, top_n: int = 12):
    grouped = (
        df.groupby(category)["Salary_USD"].mean().sort_values(ascending=False).head(top_n)
    )
    fig = px.bar(
        grouped, x=grouped.index, y=grouped.values,
        title=f"Average Salary by {category}", labels={"y": "Average Salary (USD)", "x": category},
    )
    return fig


def count_by_category_chart(df: pd.DataFrame, category: str, top_n: int = 12):
    counts = df[category].value_counts().head(top_n)
    fig = px.bar(
        counts, x=counts.index, y=counts.values,
        title=f"Job Count by {category}", labels={"y": "Number of Jobs", "x": category},
    )
    return fig


def remote_vs_onsite_chart(df: pd.DataFrame):
    def bucket(r):
        if r == 0:
            return "Onsite"
        elif r == 100:
            return "Remote"
        return "Hybrid"
    dist = df["Remote Ratio"].apply(bucket).value_counts()
    fig = px.pie(names=dist.index, values=dist.values, title="Remote vs Onsite vs Hybrid")
    return fig


def top_skills_chart(df: pd.DataFrame, top_n: int = 15):
    skills_series = df["Skills List"].explode()
    counts = skills_series.value_counts().head(top_n)
    fig = px.bar(
        counts, x=counts.values, y=counts.index, orientation="h",
        title=f"Top {top_n} Most Demanded Skills",
        labels={"x": "Number of Job Postings", "y": "Skill"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def salary_vs_experience_chart(df: pd.DataFrame):
    grouped = df.groupby("Experience Level")["Salary_USD"].mean().reindex(
        ["Entry", "Mid", "Senior", "Lead"]
    )
    fig = px.bar(
        grouped, x=grouped.index, y=grouped.values,
        title="Average Salary by Experience Level",
        labels={"y": "Average Salary (USD)", "x": "Experience Level"},
    )
    return fig


def actual_vs_predicted_chart(y_test, y_pred, model_name: str):
    fig = px.scatter(
        x=y_test, y=y_pred, opacity=0.5,
        labels={"x": "Actual Salary (USD)", "y": "Predicted Salary (USD)"},
        title=f"Actual vs Predicted Salary — {model_name}",
    )
    min_v, max_v = min(y_test), max(y_test)
    fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                  line=dict(color="red", dash="dash"))
    return fig


def feature_importance_chart(feature_importance, top_n: int = 15):
    items = feature_importance[:top_n]
    names = [f[0] for f in items][::-1]
    values = [f[1] for f in items][::-1]
    fig = px.bar(
        x=values, y=names, orientation="h",
        title="Top Feature Importances", labels={"x": "Importance", "y": "Feature"},
    )
    return fig
