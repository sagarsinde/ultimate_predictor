# System Architecture

This document serves as the living map of the `hybrid_1_1` prediction engine. It visually maps the flow of data from raw CSV files to final calibrated predictions.

## Repository Map

```mermaid
graph TD
    A[data/*.csv] -->|load_raw_data| B(hybrid_1_1/features.py)
    B -->|Feature Engineering| C(hybrid_1_1/validator.py)
    
    subgraph Walk-Forward Validation
        C --> D[Feature Ablation]
        D --> E[Train Sandboxed Models]
        E --> F[Weight Learner]
        F --> G[Model Pruning]
    end
    
    G --> H[State Management JSON]
    
    subgraph Daily Inference
        H --> I(hybrid_1_1/predict.py)
        I --> J[Tier 1: Probability Ensemble]
        
        M[panas/features.py] --> N[Tier 2: Pana Type Classifier]
        N --> O[SP/DP/TP Probabilities]
        
        J --> K[Confidence Calibration]
        K --> L((Main Digit Prediction))
        
        L --> P{Cross-Product}
        O --> P
        P --> Q((Top Recommended Panas))
    end
```

## Pipeline Components

### 1. Data Flow (`features.py`)
- Reads raw historical data from the `data/` directory.
- Constructs short-term matrices (1M, 2M, 3M) for tabular models.
- Constructs the `full` historical sequence for deep learning models (LSTMs).

### 2. Validation Pipeline (`validator.py`)
- **Feature Ablation:** Automatically toggles feature groups on and off, discarding groups that increase the Brier Score.
- **Walk-Forward Validation:** Simulates real-time betting by shifting the training window forward in time and predicting the unseen future.
- **Weight Learner:** Assigns voting power to models based on their historic Top-3 accuracy during validation.
- **Model Pruning:** Eliminates weak models that do not significantly contribute to the top 95% of cumulative probability.

### 3. Prediction Pipeline (`predict.py`)
- **Tier 1 (Main Digit):** 
  - **Probability Ensemble:** Combines the predictions from all surviving models using their learned weights.
  - **Confidence Calibration:** Maps the entropy/spread of the ensemble predictions to a confidence signal (`STRONG`, `GOOD`, `MARGINAL`, `SKIP`).
- **Tier 2 (Pana Type):**
  - **Pana Classifier (`panas/model.py`):** Trains an XGBoost classifier on historical momentum (days since last Double Pana, Single Pana runs) to predict whether tomorrow will be an SP, DP, or TP.
- **The Cross-Product:**
  - Automatically maps the top predicted Main Digit against the top predicted Pana Type to output the exact subset of recommended Panas (e.g. Double Panas that sum to 3).

### 4. State Management
- Saves the exact surviving features, weights, and calibration thresholds to `hybrid_1_1/state/` so inference scripts run instantly without retraining.
