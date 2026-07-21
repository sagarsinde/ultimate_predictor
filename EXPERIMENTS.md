# Experiments Tracker

This document logs all past experiments to prevent regression and lost knowledge.

## EXPERIMENT-001: LSTM Micro-Window Training
* **Hypothesis:** Deep learning models can capture short-term momentum better than XGBoost if trained on 1m, 2m, and 3m rolling datasets.
* **Code version:** `hybrid_1_1`
* **Dataset:** Madhur (Morning/Evening)
* **Metrics:** Top-3 Accuracy
* **Outcome:** FAILED. 
* **Lessons learned:** Deep learning models require significant data to prevent overfitting. Providing an LSTM with only 12 sequences (from a 1m window) results in catastrophic overfitting. LSTMs must be restricted to the `full` historical window.

## EXPERIMENT-002: AutoGluon Time Constraint
* **Hypothesis:** AutoGluon's `TabularPredictor` can be embedded into a Walk-Forward validation loop without hanging the system.
* **Code version:** `hybrid_1_1`
* **Dataset:** All Markets
* **Metrics:** Execution Time & Validation Brier Score
* **Outcome:** PENDING
* **Lessons learned:** A strict `time_limit` (e.g., 30s) must be enforced on `TabularPredictor.fit()` or the auto-stacking process will cause the Walk-Forward loop to run indefinitely.
