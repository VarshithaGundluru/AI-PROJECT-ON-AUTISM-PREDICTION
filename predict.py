"""Run one educational screening-score prediction from a JSON input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a dataset-derived screening flag from JSON.")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "model.joblib")
    parser.add_argument("--input", type=Path, required=True, help="JSON file containing one feature object.")
    args = parser.parse_args()

    model = joblib.load(args.model)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    row = pd.DataFrame([payload])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    print(json.dumps({
        "screening_flag": prediction,
        "screening_probability": round(probability, 4),
        "interpretation": "Dataset-derived screening proxy only; not a diagnosis or medical recommendation.",
    }, indent=2))


if __name__ == "__main__":
    main()
