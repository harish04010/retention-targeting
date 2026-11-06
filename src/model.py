"""Model comparison, final fit, and calibration."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BINARY, CATEGORICAL, FEATURES, NUMERIC

SEED = 42


def preprocessor(binary: list[str] | None = None) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("bin", "passthrough", binary if binary is not None else BINARY),
        ("num", StandardScaler(), NUMERIC),
    ])


def logistic(binary: list[str] | None = None) -> Pipeline:
    return Pipeline([("prep", preprocessor(binary)), ("model", LogisticRegression(C=0.5, max_iter=2000))])


def candidates() -> dict:
    return {
        "logistic_regression": logistic(),
        "random_forest": Pipeline([("prep", preprocessor()),
                                   ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                                                    random_state=SEED, n_jobs=-1))]),
        "gradient_boosting": Pipeline([("prep", preprocessor()),
                                       ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                                max_leaf_nodes=15, l2_regularization=1.0,
                                                                                random_state=SEED))]),
    }


def split(df: pd.DataFrame):
    return train_test_split(df, test_size=0.2, stratify=df["churned"], random_state=SEED)


def compare(train: pd.DataFrame) -> pd.DataFrame:
    """5-fold stratified CV on the training split only. The holdout stays untouched."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, pipe in candidates().items():
        res = cross_validate(pipe, train[FEATURES], train["churned"], cv=cv,
                             scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                                      "brier": "neg_brier_score"})
        rows.append({"model": name,
                     "roc_auc": res["test_roc_auc"].mean(), "roc_auc_sd": res["test_roc_auc"].std(),
                     "pr_auc": res["test_pr_auc"].mean(), "brier": -res["test_brier"].mean()})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def holdout_metrics(y, p) -> dict:
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
            "base_rate": float(np.mean(y))}


def calibration_bins(y, p, bins: int = 10) -> list[dict]:
    """Predicted vs actual churn rate by decile of predicted risk."""
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        out.append({"predicted": float(np.mean(p[chunk])), "actual": float(np.mean(np.asarray(y)[chunk])),
                    "n": int(len(chunk))})
    return out
