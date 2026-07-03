"""
Preprocessing utilities — shared by notebooks (Day 1-4) and the
Streamlit app (Day 6) so the exact same transforms are applied
at train time and inference time.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, DROP_COLS, NUMERIC_COLS, TARGET_COL


def load_raw_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges, drop nulls, encode target, drop ID col."""
    df = df.copy()

    # TotalCharges has blank strings for new customers (tenure=0) -> NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])  # ~11 rows

    # Encode target
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # Engineered feature
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: scale numeric, one-hot encode categorical."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def build_pipeline(model) -> Pipeline:
    """Full pipeline: preprocessing -> model. Use the SAME object for
    training, SHAP (via named_steps['preprocessor']), and the app."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def get_feature_names(pipeline: Pipeline) -> list:
    """Pull human-readable feature names out after one-hot encoding —
    needed to label SHAP plots correctly instead of showing x0, x1..."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
