"""
Model training/eval helpers — Day 3-4. Keeps notebook cells short and
keeps the AUC/CV logic consistent across the three models.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve


def cross_val_auc(pipeline, X, y, n_splits=5, random_state=42) -> dict:
    """5-fold stratified CV, returns mean/std AUC."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return {"mean_auc": scores.mean(), "std_auc": scores.std(), "scores": scores}


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    """Holdout AUC + ROC curve points, for plotting all 3 models together."""
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)
    return {"auc": auc, "fpr": fpr, "tpr": tpr, "probs": probs}


def compute_scale_pos_weight(y_train) -> float:
    """neg/pos ratio for XGBoost's scale_pos_weight — compute from the
    actual split rather than hardcoding 2.7, since it shifts with the seed."""
    y_train = np.asarray(y_train)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return neg / pos
