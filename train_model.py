"""Train and compare educational screening-score classifiers.

This project is not a diagnostic system. The dataset contains a continuous
``result`` score rather than a verified clinical target, so this script creates
an explicitly labeled proxy target using a configurable threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "autism.csv"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42


def load_data(path: Path, threshold: float) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load the CSV and create a transparent proxy screening target."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    frame = frame.replace({"?": np.nan, "": np.nan})

    if "result" not in frame.columns:
        raise ValueError("Expected a continuous 'result' column in the dataset.")

    frame["result"] = pd.to_numeric(frame["result"], errors="coerce")
    frame = frame.dropna(subset=["result"]).copy()
    frame["screening_flag"] = (frame["result"] >= threshold).astype(int)

    # ID and result are excluded from features. Including result would leak the
    # exact value used to construct the target into the model.
    drop_columns = ["ID", "result", "screening_flag"]
    features = frame.drop(columns=[c for c in drop_columns if c in frame.columns])
    target = frame["screening_flag"]

    metadata = {
        "rows_used": int(len(frame)),
        "feature_count": int(features.shape[1]),
        "threshold": float(threshold),
        "positive_class_count": int(target.sum()),
        "negative_class_count": int((target == 0).sum()),
        "target_definition": f"screening_flag = result >= {threshold}",
        "target_warning": "screening_flag is a dataset-derived proxy, not a clinical diagnosis.",
    }
    return features, target, metadata


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [c for c in features.columns if c.startswith("A") or c == "age"]
    categorical_columns = [c for c in features.columns if c not in numeric_columns]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_columns),
            ("categorical", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
    )


def model_candidates() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
        "support_vector_machine": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }


def evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def train(data_path: Path, artifacts_dir: Path, threshold: float) -> dict:
    features, target, metadata = load_data(data_path, threshold)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    results = {}
    fitted = {}
    for name, estimator in model_candidates().items():
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(features)),
                ("classifier", estimator),
            ]
        )
        model.fit(x_train, y_train)
        results[name] = evaluate(model, x_test, y_test)
        fitted[name] = model

    best_name = max(results, key=lambda name: (results[name]["f1"], results[name]["balanced_accuracy"]))
    best_model = fitted[best_name]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, artifacts_dir / "model.joblib")

    report = {
        "best_model": best_name,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "dataset": metadata,
        "metrics": results,
        "limitations": [
            "The target is derived from the continuous result field and is not a verified clinical label.",
            "The dataset is small and should not be used to make healthcare decisions.",
            "Model performance is a single holdout evaluation, not evidence of clinical validity.",
        ],
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train educational screening-score classifiers.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--threshold", type=float, default=6.0, help="Proxy result-score threshold; verify before use.")
    args = parser.parse_args()

    report = train(args.data, args.artifacts, args.threshold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
