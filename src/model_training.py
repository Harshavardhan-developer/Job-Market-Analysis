"""
model_training.py
------------------
Trains and compares Linear Regression, Random Forest, and Gradient Boosting
models for salary prediction, using a preprocessing pipeline (imputation +
scaling for numeric, one-hot encoding for categorical). Saves the best
model, the fitted preprocessing pipeline, evaluation metrics, feature
importance, and metadata needed for the dashboard - all to /models.

No results are hard-coded: every metric here comes from an actual
train/test split and actual fitted models.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, get_top_skills, build_feature_frame,
)

RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "jobs_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    df["Skills List"] = df["Skills List"].apply(eval) if df["Skills List"].dtype == object and \
        isinstance(df["Skills List"].iloc[0], str) else df["Skills List"]
    return df


def build_preprocessor(skill_cols):
    numeric_cols = NUMERIC_FEATURES
    categorical_cols = CATEGORICAL_FEATURES

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    # skill flag columns are already 0/1, pass through unchanged
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
        ("skill", "passthrough", skill_cols),
    ])
    return preprocessor


def get_feature_names(preprocessor, skill_cols):
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names + skill_cols


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()

    top_skills = get_top_skills(df)
    X, y, feature_cols, skill_cols = build_feature_frame(df, top_skills)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(skill_cols)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.08, random_state=RANDOM_SEED
        ),
    }

    results = {}
    fitted_pipelines = {}
    predictions = {}

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
        fitted_pipelines[name] = pipe
        predictions[name] = y_pred.tolist()

    # Select best model by R2 score
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_pipe = fitted_pipelines[best_name]

    # Feature importance (tree-based models only support this directly)
    feature_importance = None
    if best_name in ["Random Forest", "Gradient Boosting"]:
        fitted_preprocessor = best_pipe.named_steps["preprocessor"]
        feat_names = get_feature_names(fitted_preprocessor, skill_cols)
        importances = best_pipe.named_steps["model"].feature_importances_
        feature_importance = sorted(
            zip(feat_names, importances.tolist()), key=lambda x: x[1], reverse=True
        )
    else:
        # For Linear Regression, use absolute standardized coefficients as a proxy
        fitted_preprocessor = best_pipe.named_steps["preprocessor"]
        feat_names = get_feature_names(fitted_preprocessor, skill_cols)
        coefs = best_pipe.named_steps["model"].coef_
        feature_importance = sorted(
            zip(feat_names, np.abs(coefs).tolist()), key=lambda x: x[1], reverse=True
        )

    # Save artifacts
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_model.joblib"))
    for name, pipe in fitted_pipelines.items():
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join(MODELS_DIR, fname))

    metadata = {
        "best_model": best_name,
        "results": results,
        "top_skills": top_skills,
        "feature_cols": feature_cols,
        "skill_cols": skill_cols,
        "feature_importance": feature_importance[:20],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": y_test.tolist(),
        "predictions_by_model": predictions,
        "salary_std_by_segment": {
            "overall_std": float(y.std()),
        },
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model comparison:")
    for name, m in results.items():
        marker = "  <-- BEST" if name == best_name else ""
        print(f"  {name:20s} MAE={m['MAE']:>10,.2f}  RMSE={m['RMSE']:>10,.2f}  R2={m['R2']:.4f}{marker}")
    print(f"\nBest model: {best_name}")
    print(f"Artifacts saved to: {MODELS_DIR}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate()
