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

## The Problem

Workers in high-stakes environments — healthcare, aviation, logistics, emergency services — routinely make critical decisions after long shifts with little sleep. Decision fatigue is real and dangerous: as cognitive load accumulates, the quality of decisions degrades, error rates rise, and the consequences can be severe. Yet most organizations have no systematic way to measure or predict when a worker's decision-making capacity is compromised.

## What ShiftGuard Does

ShiftGuard predicts a **Decision Fatigue Score (0–100)** in real time based on a worker's current shift conditions and behavioral signals. A score closer to 100 indicates critically impaired decision-making capacity; a score under 30 means the worker is operating well. This gives supervisors and individuals an early, objective warning before fatigue turns into error.

## How It Works

A Ridge regression model was trained on 25,000 shift records containing behavioral and physiological features. Given 10 inputs, the model returns a fatigue score in milliseconds — no wearables, no lab tests, just a quick self-report form.

**Model performance (hold-out test set):**
| Metric | Value |
|--------|-------|
| R² | 0.96 |
| MAE | 6.11 |
| RMSE | 7.41 |

## Features Used

| Feature | Description |
|---------|-------------|
| Hours Awake | Total hours since last sleep |
| Sleep Hours Last Night | Hours of sleep before the shift |
| Time of Day | Morning / Afternoon / Evening / Night |
| Decisions Made | Number of decisions made during the shift |
| Task Switches | Number of times context was switched |
| Avg Decision Time (sec) | Average time taken per decision |
| Caffeine Intake (cups) | Cups of coffee/tea consumed |
| Stress Level (1–10) | Self-reported stress |
| Error Rate | Proportion of incorrect decisions |
| Cognitive Load Score | Self-reported mental load |

## Tech Stack

`Python` · `scikit-learn` · `Streamlit` · `Hugging Face Spaces`

## Live App

[https://Chris262006-shiftguard.hf.space](https://Chris262006-shiftguard.hf.space)
