# Engineering Decisions Log

Every major architectural or feature decision must be logged here to prevent repeating failed experiments.

---

```text
Decision:
Implement the Sandbox Mandate (Window Isolation)

Why it was considered:
Deep learning models (LSTM) were severely overfitting and crashing the ensemble when run on the 1m, 2m, and 3m short-term datasets.

Evidence:
A 1-month window only contains ~26 rows of training data. An LSTM requiring a 14-day lookback yields only 12 training examples, leading to catastrophic overfitting (100% training accuracy, near 0% validation accuracy).

Backtest result:
The LSTM was successfully validated when forced to train exclusively on the `full` historical window.

Approved / Rejected:
Approved

Reason:
It is mathematically impossible to train sequence models on micro-windows. Tree-based models must handle the short-term momentum, while sequence models must act as the deep-memory anchors.

Date:
2026-07-10
```

---

```text
Decision:
Introduce AutoGluon AutoML Framework

Why it was considered:
To automate hyperparameter tuning and model stacking, ensuring we don't miss optimizations that standard XGBoost/RF might miss.

Evidence:
AutoGluon is an industry standard for out-of-the-box tabular performance, wrapping multiple GBMs and Neural Nets into a single stacked ensemble.

Backtest result:
Successfully integrated into the codebase with a strict 30-second `time_limit` per fit to avoid hanging the walk-forward validator.

Approved / Rejected:
Approved

Reason:
Improves the theoretical upper bound of the tabular models without requiring manual hyperparameter sweeps for every single run.

Date:
2026-07-21
```

---

```text
Decision:
Implement Two-Tier Pana Prediction (Cross-Product)

Why it was considered:
The user wanted to predict 3-digit Panas. Predicting the exact 3-digit string out of 220 possible combinations directly using a single model is mathematically impossible and will result in 99% noise.

Evidence:
The game mathematically maps Panas to a single Main Digit via `sum(digits) % 10`. Since our existing ensemble already excels at predicting the Main Digit, we only need to predict the *Type* of Pana (Single, Double, Triple). 

Backtest result:
Pana classifier separates into SP/DP/TP correctly based on historical momentum. The cross-product mathematically filters the 220 combinations down to <10 valid recommendations.

Approved / Rejected:
Approved

Reason:
Splits a mathematically chaotic problem (1/220) into two highly constrained, solvable problems (1/10 digit prediction + 1/3 type classification). Built in a dedicated `panas/` folder to prevent cluttering the core validation loop.

Date:
2026-07-21
```
