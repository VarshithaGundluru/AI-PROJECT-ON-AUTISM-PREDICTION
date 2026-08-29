# Autism Screening Prediction — Educational Machine Learning Project

An educational machine-learning project that compares classical classifiers on an autism screening dataset. The repository contains the original dataset and reports, plus a reproducible training pipeline and a JSON prediction script.

> **Important:** This project is for learning and software demonstration only. It does not diagnose autism, assess an individual’s health, or replace a qualified healthcare professional. The model target is a dataset-derived proxy, not a verified clinical outcome.

## What this project demonstrates

- Reproducible data loading and preprocessing
- Missing-value handling and categorical encoding
- Leakage-aware feature selection
- Comparison of Logistic Regression, Decision Tree, Random Forest, and SVM models
- Holdout evaluation using accuracy, balanced accuracy, precision, recall, F1, and ROC AUC
- Persisting the best model for a JSON-based prediction example
- Clear limitations for a sensitive healthcare-adjacent use case

## Dataset

`autism.csv` contains 200 records with AQ-style question scores (`A1_Score` through `A10_Score`), demographics, screening context, and a continuous `result` field.

The CSV does **not** contain an explicit verified clinical diagnosis label. To make the classifier demonstrable, the training script creates a configurable proxy target:

```text
screening_flag = 1 when result >= threshold, otherwise 0
```

The default threshold is `6.0`. It is a modeling assumption and must be validated against an authoritative label before any serious evaluation. The continuous `result` column is excluded from features to prevent target leakage.

## Project structure

```text
.
├── autism.csv
├── Project Report.pdf
├── RESEARCH PAPER ON AUTISM PREDICTION (1).pdf
├── projectipynb.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── train_model.py
│   └── predict.py
└── artifacts/              # generated locally; ignored by Git
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train and evaluate

From the repository root:

```bash
python src/train_model.py
```

This creates:

- `artifacts/model.joblib` — the best fitted pipeline
- `artifacts/metrics.json` — model comparison metrics and dataset metadata

To use a different proxy threshold:

```bash
python src/train_model.py --threshold 7.0
```

The script uses a fixed random seed (`42`) and stratified holdout split so the run is reproducible.

## Run a prediction

Create a JSON file containing the feature fields used by the model, for example:

```json
{
  "A1_Score": 1,
  "A2_Score": 0,
  "A3_Score": 1,
  "A4_Score": 0,
  "A5_Score": 1,
  "A6_Score": 0,
  "A7_Score": 1,
  "A8_Score": 0,
  "A9_Score": 1,
  "A10_Score": 0,
  "age": 25,
  "gender": "f",
  "ethnicity": "White-European",
  "jaundice": "no",
  "austim": "no",
  "contry_of_res": "India",
  "used_app_before": "no",
  "age_desc": "18 and more",
  "relation": "Self"
}
```

Then run:

```bash
python src/predict.py --input sample_input.json
```

The output is a probability for the dataset-derived screening proxy. It must not be interpreted as a diagnosis.

## Original materials

The repository retains the original `projectipynb.ipynb`, dataset, project report, and research paper. The new `src/` pipeline is intentionally separate so the original work remains available for comparison.

## Limitations and responsible-use notes

- The dataset is small and may not represent the wider population.
- Demographic and geographic fields can encode sampling bias or historical bias.
- The proxy target is derived from `result`, not a verified clinical diagnosis.
- Accuracy alone is not sufficient for a sensitive screening use case; calibration, subgroup analysis, external validation, and clinical review would be required.
- This code should not be used for diagnosis, triage, treatment, or decisions about a person.

## Future improvements

1. Replace the proxy target with a documented, ethically sourced, clinically validated label.
2. Add cross-validation, calibration curves, confusion matrices, and subgroup fairness checks.
3. Track dataset provenance, consent, governance, and model-card documentation.
4. Add tests for preprocessing, schema validation, and prediction input handling.
5. Keep a human expert in the loop for any research evaluation.

## Author

Varshitha Gundluru
