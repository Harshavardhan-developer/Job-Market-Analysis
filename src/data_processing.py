"""
data_processing.py
-------------------
Handles:
  1. Synthetic dataset generation (since no reliable public dataset is bundled
     with this environment, a realistic synthetic dataset is generated
     programmatically instead - this is clearly disclosed here and in the README).
  2. Data cleaning: missing values, duplicates, invalid salaries, outliers,
     salary normalization, location/job-title standardization, encoding.

Run this file directly to regenerate data/raw/jobs_raw.csv and
data/processed/jobs_clean.csv from scratch.
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "jobs_raw.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "jobs_clean.csv")

# --------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# --------------------------------------------------------------------------

JOB_TITLES = [
    "Data Scientist", "Data Analyst", "Machine Learning Engineer",
    "Software Engineer", "Backend Developer", "Frontend Developer",
    "Full Stack Developer", "Data Engineer", "AI Research Scientist",
    "Business Analyst", "DevOps Engineer", "BI Analyst",
    "Data Architect", "NLP Engineer", "Computer Vision Engineer",
]

# Base salary (USD/year, "mid" experience) used as the seed for simulation.
JOB_BASE_SALARY = {
    "Data Scientist": 95000, "Data Analyst": 65000,
    "Machine Learning Engineer": 110000, "Software Engineer": 90000,
    "Backend Developer": 85000, "Frontend Developer": 80000,
    "Full Stack Developer": 88000, "Data Engineer": 100000,
    "AI Research Scientist": 125000, "Business Analyst": 68000,
    "DevOps Engineer": 98000, "BI Analyst": 70000,
    "Data Architect": 118000, "NLP Engineer": 112000,
    "Computer Vision Engineer": 113000,
}

LOCATIONS = [
    "New York, USA", "San Francisco, USA", "Austin, USA", "Seattle, USA",
    "London, UK", "Berlin, Germany", "Toronto, Canada", "Bangalore, India",
    "Hyderabad, India", "Singapore", "Sydney, Australia", "Dublin, Ireland",
    "Remote",
]

# Cost-of-living style multiplier applied to base salary by location.
LOCATION_MULTIPLIER = {
    "New York, USA": 1.35, "San Francisco, USA": 1.5, "Austin, USA": 1.1,
    "Seattle, USA": 1.3, "London, UK": 1.15, "Berlin, Germany": 0.95,
    "Toronto, Canada": 1.05, "Bangalore, India": 0.35, "Hyderabad, India": 0.32,
    "Singapore": 1.1, "Sydney, Australia": 1.1, "Dublin, Ireland": 1.05,
    "Remote": 1.0,
}

EXPERIENCE_LEVELS = ["Entry", "Mid", "Senior", "Lead"]
EXPERIENCE_MULTIPLIER = {"Entry": 0.65, "Mid": 1.0, "Senior": 1.45, "Lead": 1.9}

EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Contract", "Internship"]
EMPLOYMENT_MULTIPLIER = {"Full-time": 1.0, "Part-time": 0.55, "Contract": 0.9, "Internship": 0.35}

INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "E-commerce", "Consulting",
    "Education", "Manufacturing", "Telecommunications", "Media", "Government",
]
INDUSTRY_MULTIPLIER = {
    "Technology": 1.15, "Finance": 1.2, "Healthcare": 1.0, "E-commerce": 1.05,
    "Consulting": 1.1, "Education": 0.85, "Manufacturing": 0.9,
    "Telecommunications": 0.95, "Media": 0.9, "Government": 0.8,
}

COMPANY_SIZES = ["Small", "Medium", "Large"]
COMPANY_SIZE_MULTIPLIER = {"Small": 0.9, "Medium": 1.0, "Large": 1.15}

EDUCATION_LEVELS = ["High School", "Bachelor", "Master", "PhD"]
EDUCATION_MULTIPLIER = {"High School": 0.85, "Bachelor": 1.0, "Master": 1.15, "PhD": 1.3}

SKILL_POOL = [
    "Python", "SQL", "Java", "JavaScript", "AWS", "Machine Learning", "Excel",
    "Power BI", "React", "TensorFlow", "PyTorch", "Docker", "Kubernetes",
    "Spark", "Tableau", "R", "C++", "NLP", "Deep Learning", "Git",
    "Azure", "GCP", "Airflow", "Statistics", "Linux",
]

COMPANIES = [
    "NexaData", "CloudForge", "Brightline Analytics", "Vertex Labs",
    "Quantum Softworks", "Summit Technologies", "Pinnacle Systems",
    "Northwind Digital", "BlueOrbit AI", "Fusion Insights", "Skyline Tech",
    "Orbital Data Co", "Clearwater Solutions", "Ironclad Systems", "Zenith AI",
]

CURRENCIES_BY_LOCATION = {
    "New York, USA": "USD", "San Francisco, USA": "USD", "Austin, USA": "USD",
    "Seattle, USA": "USD", "London, UK": "GBP", "Berlin, Germany": "EUR",
    "Toronto, Canada": "CAD", "Bangalore, India": "INR", "Hyderabad, India": "INR",
    "Singapore": "SGD", "Sydney, Australia": "AUD", "Dublin, Ireland": "EUR",
    "Remote": "USD",
}

# Rough FX rates to USD, used ONLY to normalize salary for modeling purposes.
FX_TO_USD = {"USD": 1.0, "GBP": 1.27, "EUR": 1.08, "CAD": 0.73,
             "INR": 0.012, "SGD": 0.74, "AUD": 0.66}


def generate_synthetic_dataset(n_rows: int = 6000) -> pd.DataFrame:
    """Generate a realistic synthetic job-market dataset.

    NOTE: This is SYNTHETIC data generated programmatically for demonstration
    purposes because no verified public dataset was bundled in this
    environment. Salaries are simulated using base-salary anchors per role,
    combined with multiplicative effects for experience, location, industry,
    company size, education and employment type, plus random noise - this
    mimics realistic labor-market patterns without claiming to be real data.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for _ in range(n_rows):
        title = rng.choice(JOB_TITLES)
        exp_level = rng.choice(EXPERIENCE_LEVELS, p=[0.30, 0.35, 0.25, 0.10])
        # years of experience consistent with level
        yoe_ranges = {"Entry": (0, 2), "Mid": (2, 5), "Senior": (5, 10), "Lead": (8, 20)}
        lo, hi = yoe_ranges[exp_level]
        years_exp = int(rng.integers(lo, hi + 1))

        location = rng.choice(LOCATIONS)
        employment_type = rng.choice(EMPLOYMENT_TYPES, p=[0.78, 0.07, 0.10, 0.05])
        industry = rng.choice(INDUSTRIES)
        company_size = rng.choice(COMPANY_SIZES, p=[0.35, 0.40, 0.25])
        education = rng.choice(EDUCATION_LEVELS, p=[0.05, 0.55, 0.32, 0.08])
        remote_ratio = int(rng.choice([0, 50, 100], p=[0.45, 0.25, 0.30]))
        company = rng.choice(COMPANIES)

        n_skills = int(rng.integers(3, 8))
        skills = list(rng.choice(SKILL_POOL, size=n_skills, replace=False))

        currency = CURRENCIES_BY_LOCATION[location]

        base = JOB_BASE_SALARY[title]
        salary = (
            base
            * LOCATION_MULTIPLIER[location]
            * EXPERIENCE_MULTIPLIER[exp_level]
            * EMPLOYMENT_MULTIPLIER[employment_type]
            * INDUSTRY_MULTIPLIER[industry]
            * COMPANY_SIZE_MULTIPLIER[company_size]
            * EDUCATION_MULTIPLIER[education]
        )
        # small bump for extra years within a level + skill count
        salary *= (1 + 0.015 * years_exp)
        salary *= (1 + 0.01 * n_skills)
        # random noise (log-normal-ish multiplicative noise)
        salary *= rng.normal(loc=1.0, scale=0.08)
        # convert to local currency for the raw "as posted" value
        salary_local = salary / FX_TO_USD[currency]

        # inject some messiness intentionally (to be cleaned later)
        row = {
            "Job Title": title if rng.random() > 0.03 else title.lower(),
            "Company": company,
            "Location": location if rng.random() > 0.03 else location.upper(),
            "Experience Level": exp_level,
            "Years of Experience": years_exp,
            "Employment Type": employment_type,
            "Remote Ratio": remote_ratio,
            "Industry": industry,
            "Company Size": company_size,
            "Education Level": education,
            "Skills": ", ".join(skills),
            "Salary": round(salary_local, 2),
            "Currency": currency,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Inject missing values (~3% of cells across a few columns)
    for col in ["Education Level", "Industry", "Company Size", "Years of Experience", "Skills"]:
        mask = rng.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    # Inject some duplicate rows (~1.5%)
    dup_frac = 0.015
    dup_rows = df.sample(frac=dup_frac, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Inject a handful of invalid salaries (negative / zero / absurdly high)
    n_invalid = max(5, int(0.005 * len(df)))
    invalid_idx = rng.choice(df.index, size=n_invalid, replace=False)
    for i, idx in enumerate(invalid_idx):
        if i % 2 == 0:
            df.loc[idx, "Salary"] = -1 * abs(df.loc[idx, "Salary"])
        else:
            df.loc[idx, "Salary"] = df.loc[idx, "Salary"] * 50  # unrealistic outlier

    return df


# --------------------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------------------

def standardize_location(loc: str) -> str:
    if pd.isna(loc):
        return np.nan
    loc = str(loc).strip()
    # Title-case each comma separated part for consistency (fixes ALL CAPS entries)
    parts = [p.strip() for p in loc.split(",")]
    parts = [p if p.upper() == "USA" or p.upper() == "UK" else p.title() for p in parts]
    parts = ["USA" if p.upper() == "USA" else ("UK" if p.upper() == "UK" else p) for p in parts]
    return ", ".join(parts)


def standardize_job_title(title: str) -> str:
    if pd.isna(title):
        return np.nan
    return str(title).strip().title()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full data-cleaning pipeline and return a clean DataFrame."""
    df = df.copy()

    # --- Duplicate removal ---
    before = len(df)
    df = df.drop_duplicates()
    dup_removed = before - len(df)

    # --- Standardize text fields ---
    df["Job Title"] = df["Job Title"].apply(standardize_job_title)
    df["Location"] = df["Location"].apply(standardize_location)

    # --- Invalid salary detection/removal (negative or zero) ---
    before = len(df)
    df = df[df["Salary"] > 0]
    invalid_removed = before - len(df)

    # --- Normalize salary to USD/year for consistent modeling ---
    df["Salary_USD"] = df.apply(
        lambda r: r["Salary"] * FX_TO_USD.get(r["Currency"], 1.0), axis=1
    )

    # --- Outlier detection using IQR on normalized salary ---
    q1, q3 = df["Salary_USD"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    before = len(df)
    df = df[(df["Salary_USD"] >= lower_bound) & (df["Salary_USD"] <= upper_bound)]
    outliers_removed = before - len(df)

    # --- Missing value handling ---
    # Numeric: median impute
    df["Years of Experience"] = df["Years of Experience"].fillna(
        df["Years of Experience"].median()
    )
    # Categorical: mode impute
    for col in ["Education Level", "Industry", "Company Size"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    # Skills: fill with "Not Specified"
    df["Skills"] = df["Skills"].fillna("Not Specified")

    # --- Experience-level encoding (ordinal) ---
    exp_order = {"Entry": 0, "Mid": 1, "Senior": 2, "Lead": 3}
    df["Experience Level Encoded"] = df["Experience Level"].map(exp_order)

    edu_order = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    df["Education Level Encoded"] = df["Education Level"].map(edu_order)

    # --- Skill preprocessing: clean list representation ---
    df["Skills List"] = df["Skills"].apply(
        lambda s: [sk.strip() for sk in str(s).split(",")] if s != "Not Specified" else []
    )
    df["Skill Count"] = df["Skills List"].apply(len)

    df = df.reset_index(drop=True)

    cleaning_report = {
        "duplicates_removed": int(dup_removed),
        "invalid_salaries_removed": int(invalid_removed),
        "outliers_removed": int(outliers_removed),
        "final_row_count": int(len(df)),
    }
    return df, cleaning_report


def run_pipeline(n_rows: int = 6000, force_regenerate: bool = False):
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    if force_regenerate or not os.path.exists(RAW_PATH):
        raw_df = generate_synthetic_dataset(n_rows)
        raw_df.to_csv(RAW_PATH, index=False)
    else:
        raw_df = pd.read_csv(RAW_PATH)

    clean_df, report = clean_dataset(raw_df)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    print("Data cleaning report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"Raw dataset saved to:      {RAW_PATH}  ({len(raw_df)} rows)")
    print(f"Processed dataset saved to: {PROCESSED_PATH} ({len(clean_df)} rows)")
    return clean_df, report


if __name__ == "__main__":
    run_pipeline(n_rows=6000, force_regenerate=True)
