"""
Train a regression model to predict Decision_Fatigue_Score from behavioral features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Hours_Awake",
    "Decisions_Made",
    "Task_Switches",
    "Avg_Decision_Time_sec",
    "Sleep_Hours_Last_Night",
    "Caffeine_Intake_Cups",
    "Stress_Level_1_10",
    "Error_Rate",
    "Cognitive_Load_Score",
]
CATEGORICAL_FEATURES = ["Time_of_Day"]
TARGET = "Decision_Fatigue_Score"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            ("reg", Ridge(alpha=1.0)),
        ]
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    if any(missing.values()):
        raise ValueError(f"Missing values found: {missing}")
    for col in NUMERIC_FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fatigue score regressor")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "human_decision_fatigue_dataset.csv",
        help="Path to CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "fatigue_regressor.joblib",
        help="Where to save the trained pipeline",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = load_data(args.data)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)

    meta = {
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "target": TARGET,
        "features": feature_cols,
        "model_path": str(args.out.resolve()),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model to {args.out}")
    print(f"Metrics (hold-out): MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
