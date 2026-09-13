"""
db.py
-----
Optional PostgreSQL persistence layer for job listings, salary info, skills,
and prediction history. The rest of the pipeline (data cleaning, EDA,
training, dashboard) works fully without this module or a running database -
import and call these functions only if you want to persist data.

Connection settings are read from environment variables (see .env.example).
Requires: psycopg2-binary, python-dotenv (both listed in requirements.txt).
"""

import os
import json

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:  # psycopg2 is optional - only needed if DB features are used
    psycopg2 = None

from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "job_market_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS job_listings (
    id SERIAL PRIMARY KEY,
    job_title TEXT,
    company TEXT,
    location TEXT,
    experience_level TEXT,
    years_of_experience INT,
    employment_type TEXT,
    remote_ratio INT,
    industry TEXT,
    company_size TEXT,
    education_level TEXT,
    skills TEXT,
    salary NUMERIC,
    currency TEXT,
    salary_usd NUMERIC
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id SERIAL PRIMARY KEY,
    input_payload JSONB,
    predicted_salary NUMERIC,
    salary_range_low NUMERIC,
    salary_range_high NUMERIC,
    model_used TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
"""


def get_connection():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Run: pip install psycopg2-binary")
    return psycopg2.connect(**DB_CONFIG)


def init_schema():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def load_job_listings_from_csv(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO job_listings
                    (job_title, company, location, experience_level, years_of_experience,
                     employment_type, remote_ratio, industry, company_size, education_level,
                     skills, salary, currency, salary_usd)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        row.get("Job Title"), row.get("Company"), row.get("Location"),
                        row.get("Experience Level"), int(row.get("Years of Experience", 0)),
                        row.get("Employment Type"), int(row.get("Remote Ratio", 0)),
                        row.get("Industry"), row.get("Company Size"), row.get("Education Level"),
                        row.get("Skills"), float(row.get("Salary", 0)), row.get("Currency"),
                        float(row.get("Salary_USD", 0)),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def save_prediction(input_payload: dict, predicted_salary: float, salary_range, model_used: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_history
                (input_payload, predicted_salary, salary_range_low, salary_range_high, model_used)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (Json(input_payload), predicted_salary, salary_range[0], salary_range[1], model_used),
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    print("Initializing schema (requires a running PostgreSQL instance)...")
    init_schema()
    print("Schema ready.")
