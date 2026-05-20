---
title: ShiftGuard
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# ShiftGuard — Decision Fatigue Score Predictor

Predicts a **Decision_Fatigue_Score (0–100)** from 10 behavioral and environmental features using a Ridge regression model trained on 25,000 shift records.

**Model metrics (hold-out test set):** MAE = 6.11 · RMSE = 7.41 · R² = 0.96

## Features used
- Hours Awake
- Decisions Made
- Task Switches
- Avg Decision Time (sec)
- Sleep Hours Last Night
- Time of Day (Morning / Afternoon / Evening / Night)
- Caffeine Intake (cups)
- Stress Level (1–10)
- Error Rate
- Cognitive Load Score
