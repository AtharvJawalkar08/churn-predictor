import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.modeling import cross_val_auc, evaluate_on_test, compute_scale_pos_weight
from src.preprocessing import load_raw_data, clean_data, build_pipeline
from src.config import TARGET_COL

DATA_PATH = "data/telco_churn.csv"

def main():
    mlflow.set_experiment("churn-predictor")

    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = compute_scale_pos_weight(y_train)

    with mlflow.start_run():
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        cv_results = cross_val_auc(pipeline, X_train, y_train)
        test_results = evaluate_on_test(pipeline, X_test, y_test)

        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metric("cv_mean_auc", cv_results["mean_auc"])
        mlflow.log_metric("cv_std_auc", cv_results["std_auc"])
        mlflow.log_metric("test_auc", test_results["auc"])

        mlflow.sklearn.log_model(pipeline, "model", serialization_format="cloudpickle")

        print(f"CV AUC: {cv_results['mean_auc']:.4f} +/- {cv_results['std_auc']:.4f}")
        print(f"Test AUC: {test_results['auc']:.4f}")

if __name__ == "__main__":
    main()
