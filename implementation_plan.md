# Multi-Window Consensus Prediction Engine v2

## Problem
The current system trains ONE XGBoost model on ALL historical data (2013-2025). Old patterns dilute recent trends, resulting in weak predictions. We need a smarter system that captures patterns at multiple time scales and only bets when multiple independent signals agree.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              RAW DATASET (Kalyan / MB)               │
└──────────┬──────────┬──────────┬──────────┬─────────┘
           │          │          │          │
     ┌─────▼────┐┌────▼─────┐┌──▼───────┐┌▼────────┐
     │ 1 Month  ││ 2 Months ││ 3 Months ││  Full   │
     │ Window   ││ Window   ││ Window   ││ History │
     └─────┬────┘└────┬─────┘└──┬───────┘└┬────────┘
           │          │          │          │
     ┌─────▼────┐┌────▼─────┐┌──▼───────┐┌▼────────┐
     │ XGBoost  ││ XGBoost  ││ XGBoost  ││ XGBoost │
     │ RandForst││ RandForst││ RandForst││ RandForst│
     │ Markov   ││ Markov   ││ Markov   ││ Markov  │
     │ Frequency││ Frequency││ Frequency││Frequency│
     └─────┬────┘└────┬─────┘└──┬───────┘└┬────────┘
           │          │          │          │
           └──────────┴──────┬───┴──────────┘
                             │
                   ┌─────────▼─────────┐
                   │  CONSENSUS ENGINE  │
                   │  Score & Rank All  │
                   │  Digits 0-9        │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │  BET STRENGTH      │
                   │  🔥 STRONG / ✅ GOOD │
                   │  ⚠️ WEAK / ❌ SKIP  │
                   └───────────────────┘
```

---

## Phase 1: Multi-Window Feature Builder

### File: `build_features_v2.py`

Takes any dataset + a window parameter and builds features from ONLY that slice of data.

**Windows:**
| Window | Data Used | Purpose |
|--------|-----------|---------|
| 1 month | Last ~26 draws (Kalyan) / ~22 draws (MB) | Current hot streaks, immediate momentum |
| 2 months | Last ~52 draws / ~44 draws | Medium-term cycles |
| 3 months | Last ~78 draws / ~66 draws | Seasonal patterns |
| Full | All data | Long-term base rates |

**Features per window** (same as current, but calculated only within the window):
- Lag features (last 1-7 results)
- Streak counters (consecutive appearances)
- Gambler's Fallacy scores (how "overdue" each digit is)
- Day-of-week encoding
- Gap analysis (days since each digit last appeared)
- **NEW: Morning→Evening correlation** (if morning was X, what does evening tend to be?)
- **NEW: Hot/Cold ratio** (appearance frequency in recent N draws vs expected 10%)

---

## Phase 2: Multi-Model Trainer

### File: `train_models_v2.py`

For EACH window, trains 4 different model types:

| Model | Why |
|-------|-----|
| **XGBoost** | Best for structured tabular data, handles feature interactions |
| **Random Forest** | More robust to noise, less overfitting on small windows |
| **Markov Chain** | Pure transition probability — "what digit tends to follow what?" |
| **Frequency** | Simple hot/cold digit counting — surprisingly powerful on short windows |

**Total models per market:** 4 windows × 4 model types = **16 models**

Each model outputs a probability distribution over digits 0-9 for both Morning and Evening.

**Saved files:**
- `models_v2/kalyan_1m_xgb.joblib`, `models_v2/kalyan_1m_rf.joblib`, etc.
- `models_v2/kalyan_1m_markov.json`, `models_v2/kalyan_1m_freq.json`, etc.

---

## Phase 3: Consensus Prediction Engine

### File: `predict_v2.py`

**Step 1:** Run all 16 models. Each produces a Top 3 for Morning and Evening.

**Step 2:** Score each digit using a weighted point system:
- Ranked #1 in a model's Top 3 → **3 points**
- Ranked #2 → **2 points**
- Ranked #3 → **1 point**

**Step 3:** Sum points across all 16 models. Maximum possible score = 16 × 3 = 48 points.

**Step 4:** Determine bet strength:

| Total Score | Strength | Action |
|-------------|----------|--------|
| 30+ points | 🔥 **STRONG BET** | High confidence, multiple models and windows agree |
| 20-29 points | ✅ **GOOD BET** | Decent agreement across windows |
| 12-19 points | ⚠️ **WEAK** | Limited agreement, proceed with caution |
| Below 12 | ❌ **SKIP** | No consensus, save your money |

**Step 5:** Show breakdown table:
```
KALYAN PREDICTION: 2026-07-02 (Thursday)

MORNING CONSENSUS:
  Digit │ 1M │ 2M │ 3M │ Full │ Total │ Strength
  ──────┼────┼────┼────┼──────┼───────┼──────────
    8   │ 9  │ 8  │ 7  │  5   │  29   │ ✅ GOOD
    5   │ 6  │ 5  │ 6  │  4   │  21   │ ✅ GOOD
    3   │ 3  │ 4  │ 2  │  6   │  15   │ ⚠️ WEAK

EVENING CONSENSUS:
  Digit │ 1M │ 2M │ 3M │ Full │ Total │ Strength
  ──────┼────┼────┼────┼──────┼───────┼──────────
    6   │ 11 │ 9  │ 8  │  7   │  35   │ 🔥 STRONG
    ...
```

This way you can see EXACTLY which windows are voting for which digits and make an informed decision.

---

## Phase 4: Backtester

### File: `backtest_v2.py`

**Method (as you specified):**
1. Train all models on data up to May 31, 2026
2. Predict June 1-7 (first 7 playing days of June)
3. Compare predictions against actual results
4. Report hit rates per model type, per window, and for the consensus

**Output:**
```
BACKTEST RESULTS: Predicting June 1-7 using May data
═══════════════════════════════════════════════════════
                    Morning Hit Rate    Evening Hit Rate
  1-Month XGBoost:     3/7 (43%)          2/7 (29%)
  2-Month XGBoost:     2/7 (29%)          3/7 (43%)
  ...
  CONSENSUS (Top 1):   4/7 (57%)          4/7 (57%)
  CONSENSUS (Top 3):   6/7 (86%)          5/7 (71%)
═══════════════════════════════════════════════════════
  Random baseline:     1/7 (14%)          1/7 (14%)
```

This proves whether the system actually works before you bet real money.

---

## File Structure

```
f:\ultimate_preducter\
├── build_features_v2.py      # Multi-window feature builder
├── train_models_v2.py        # Trains 16 models (4 windows × 4 types)
├── predict_v2.py             # Consensus engine with bet strength
├── backtest_v2.py            # Backtest on May→June data
├── models_v2/                # All saved model files
│   ├── kalyan_1m_xgb.joblib
│   ├── kalyan_1m_rf.joblib
│   ├── kalyan_1m_markov.json
│   ├── ...
│   ├── mb_1m_xgb.joblib
│   └── ...
├── Prediction_Engine.ipynb   # Updated Colab notebook
├── true_kalyan_main_dataset.csv
└── main_bazar_dataset.csv
```

---

## Workflow (Daily Usage)

1. Update CSVs with yesterday's real results
2. Push to GitHub
3. In Colab notebook:
   - Cell 1: `git pull`
   - Cell 2: `!python train_models_v2.py kalyan` (trains all 16 models)
   - Cell 3: `!python predict_v2.py kalyan` (shows consensus prediction)
   - Cell 4: `!python train_models_v2.py mb`
   - Cell 5: `!python predict_v2.py mb`

---

## Open Questions

> [!IMPORTANT]
> **Scoring weights:** Should all 4 model types (XGBoost, RandomForest, Markov, Frequency) have equal weight? Or should we weight XGBoost and RandomForest higher since they're ML models?

> [!IMPORTANT]
> **Window weights:** Should all 4 windows have equal weight? Or should recent windows (1-month, 2-month) count more than full history?

> [!NOTE]
> The old v1 scripts (`build_features.py`, `train_model.py`, `predict_tomorrow.py`, etc.) will remain untouched. The new v2 system is completely separate, so you can compare them side by side.
