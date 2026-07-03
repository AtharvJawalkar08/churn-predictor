"""
Shared config — column groupings, paths, and constants used across
notebooks, src/, and app/ so nothing gets redefined inconsistently.
"""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_DIR / "telco_churn.csv"
PIPELINE_PATH = MODEL_DIR / "xgb_churn_pipeline.joblib"

# Target
TARGET_COL = "Churn"

# Columns dropped before modeling (identifiers, not predictive)
DROP_COLS = ["customerID"]

# Numeric columns (after TotalCharges is coerced to numeric)
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]

# Categorical columns to one-hot encode
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
