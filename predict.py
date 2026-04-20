"""
Load the trained regression pipeline and predict Decision_Fatigue_Score for new rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

# Must match train.py
EXPECTED_COLUMNS = [
    "Hours_Awake",
    "Decisions_Made",
    "Task_Switches",
    "Avg_Decision_Time_sec",
    "Sleep_Hours_Last_Night",
    "Time_of_Day",
    "Caffeine_Intake_Cups",
    "Stress_Level_1_10",
    "Error_Rate",
    "Cognitive_Load_Score",
]


def load_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Run train.py first."
        )
    return joblib.load(path)


def predict_row(model, row: dict) -> float:
    missing = [c for c in EXPECTED_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    X = pd.DataFrame([{k: row[k] for k in EXPECTED_COLUMNS}])
    score = float(model.predict(X)[0])
    return max(0.0, min(100.0, score))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fatigue score")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "fatigue_regressor.joblib",
        help="Path to saved pipeline from train.py",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="JSON file with one object (keys = feature names)",
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.json:
        row = json.loads(args.json.read_text(encoding="utf-8"))
    else:
        example = {
            "Hours_Awake": 10,
            "Decisions_Made": 50,
            "Task_Switches": 15,
            "Avg_Decision_Time_sec": 3.0,
            "Sleep_Hours_Last_Night": 6.0,
            "Time_of_Day": "Afternoon",
            "Caffeine_Intake_Cups": 2,
            "Stress_Level_1_10": 3.0,
            "Error_Rate": 0.02,
            "Cognitive_Load_Score": 4.0,
        }
        row = example
        print("(No --json provided; using built-in example row)\n")

    pred = predict_row(model, row)
    print(f"Predicted Decision_Fatigue_Score: {pred:.2f}")


if __name__ == "__main__":
    main()
