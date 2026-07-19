from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.modeling import cross_val_auc, compute_scale_pos_weight
from src.preprocessing import load_raw_data, clean_data, build_pipeline
from src.config import TARGET_COL

def test_model_meets_auc_threshold():
    df = load_raw_data("data/telco_churn.csv")
    df = clean_data(df)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scale_pos_weight = compute_scale_pos_weight(y_train)
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42)
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    results = cross_val_auc(pipeline, X_train, y_train)
    assert results["mean_auc"] > 0.75, f"AUC too low: {results['mean_auc']}"