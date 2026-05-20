"""
ShiftGuard — Decision Fatigue Score Predictor
Streamlit frontend for the fatigue_regressor model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "fatigue_regressor.joblib"

FEATURE_ORDER = [
    "Hours_Awake",
    "Decisions_Made",
    "Task_Switches",
    "Avg_Decision_Time_sec",
    "Sleep_Hours_Last_Night",
    "Caffeine_Intake_Cups",
    "Stress_Level_1_10",
    "Error_Rate",
    "Cognitive_Load_Score",
    "Time_of_Day",
]


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(MODEL_PATH)


def fatigue_label(score: float) -> tuple[str, str]:
    if score < 30:
        return "Low", "success"
    if score < 60:
        return "Moderate", "warning"
    if score < 80:
        return "High", "error"
    return "Critical", "error"


def main() -> None:
    st.set_page_config(
        page_title="ShiftGuard · Fatigue Predictor",
        page_icon="🛡️",
        layout="centered",
    )

    st.title("🛡️ ShiftGuard")
    st.caption("Decision Fatigue Score Predictor — powered by a Ridge regression model (R² = 0.96)")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Shift Conditions")
        hours_awake = st.slider("Hours Awake", 1, 17, 9)
        sleep_hours = st.slider("Sleep Hours Last Night", 2.0, 9.0, 6.0, step=0.1)
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        caffeine_cups = st.slider("Caffeine Intake (cups)", 0, 6, 2)

    with col_b:
        st.subheader("Work Load")
        decisions_made = st.number_input("Decisions Made", min_value=0, max_value=120, value=45)
        task_switches = st.number_input("Task Switches", min_value=0, max_value=50, value=13)
        avg_decision_time = st.number_input(
            "Avg Decision Time (sec)", min_value=0.5, max_value=7.0, value=3.0, step=0.1
        )
        error_rate = st.number_input(
            "Error Rate (0–0.36)", min_value=0.00, max_value=0.36, value=0.02, step=0.01, format="%.2f"
        )

    st.subheader("Self-Assessment")
    col_c, col_d = st.columns(2)
    with col_c:
        stress_level = st.slider("Stress Level (1–10)", 1.0, 9.0, 3.0, step=0.1)
    with col_d:
        cognitive_load = st.slider("Cognitive Load Score (0.6–9.7)", 0.6, 9.7, 3.0, step=0.1)

    st.divider()

    if st.button("Predict Fatigue Score", type="primary", use_container_width=True):
        model = load_model()

        row = {
            "Hours_Awake": hours_awake,
            "Decisions_Made": decisions_made,
            "Task_Switches": task_switches,
            "Avg_Decision_Time_sec": avg_decision_time,
            "Sleep_Hours_Last_Night": sleep_hours,
            "Caffeine_Intake_Cups": caffeine_cups,
            "Stress_Level_1_10": stress_level,
            "Error_Rate": error_rate,
            "Cognitive_Load_Score": cognitive_load,
            "Time_of_Day": time_of_day,
        }
        X = pd.DataFrame([row])[FEATURE_ORDER]
        score = float(model.predict(X)[0])
        score = max(0.0, min(100.0, score))

        label, level = fatigue_label(score)

        st.metric("Decision Fatigue Score", f"{score:.1f} / 100", delta=None)
        st.progress(score / 100)

        messages = {
            "Low": "You're operating well. Keep it up.",
            "Moderate": "Moderate fatigue detected — consider a short break soon.",
            "High": "High fatigue — reduce decision load and rest when possible.",
            "Critical": "Critical fatigue — step back from high-stakes decisions immediately.",
        }
        getattr(st, level)(f"**{label} fatigue.** {messages[label]}")

        with st.expander("Input summary"):
            st.json(row)


if __name__ == "__main__":
    main()
