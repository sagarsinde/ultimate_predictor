import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def main():
    print("--- Phase 2: Professional Training Engine ---")
    
    # 1. Load Data Lake
    df = pd.read_csv('kalyan_feature_store.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Strict Date Filtering (The 2026 Vault)
    train_df = df[df['Date'] < '2026-01-01'].copy()
    test_df = df[df['Date'] >= '2026-01-01'].copy() # Just for logging
    
    print(f"Total Rows Loaded: {len(df)}")
    print(f"Training Set (2013-2025): {len(train_df)} rows")
    print(f"Hold-Out Vault (2026): {len(test_df)} rows securely locked.")
    
    # 3. Define Features
    drop_cols = ['Date', 'Day_of_Week', 'Jodi', 'Morning_number', 'Evening_number', 'Draw_Index', 
                 'Morning_card1', 'Morning_card2', 'Morning_card3', 
                 'Evening_number1', 'Evening_number2', 'Evening_number3']
    
    features = [c for c in train_df.columns if c not in drop_cols]
    
    X_train = train_df[features]
    y_m_train = train_df['Morning_number'].astype(int)
    y_e_train = train_df['Evening_number'].astype(int)
    
    # 4. Define the Hardware-Accelerated Model
    print("\nInitializing GPU-Accelerated XGBoost engines...")
    params = {
        'objective': 'multi:softprob',
        'num_class': 10,
        'eval_metric': 'mlogloss',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 150,
        'tree_method': 'hist',
        'device': 'cuda', # Must run on Colab T4 GPU
        'random_state': 42
    }
    
    base_xgb_m = xgb.XGBClassifier(**params)
    base_xgb_e = xgb.XGBClassifier(**params)
    
    # 5. Isotonic Calibration with Walk-Forward CV
    print("Configuring Isotonic Calibrator with TimeSeriesSplit (5 Folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    calibrated_m = CalibratedClassifierCV(estimator=base_xgb_m, method='isotonic', cv=tscv)
    calibrated_e = CalibratedClassifierCV(estimator=base_xgb_e, method='isotonic', cv=tscv)
    
    # 6. Fit Models (Heavy Computation)
    print("\nCommencing Training Phase (This will take a moment on the GPU)...")
    print("Training Morning Model...")
    calibrated_m.fit(X_train, y_m_train)
    
    print("Training Evening Model...")
    calibrated_e.fit(X_train, y_e_train)
    
    # 7. Serialization (Saving the Brains)
    print("\nSerialization Protocol Engaged...")
    joblib.dump(calibrated_m, 'morning_model.joblib')
    joblib.dump(calibrated_e, 'evening_model.joblib')
    
    with open('model_features.json', 'w') as f:
        json.dump(features, f)
        
    print("SUCCESS: Master models saved to disk (morning_model.joblib, evening_model.joblib)")
    print("SUCCESS: Feature map saved to model_features.json")
    print("\nPhase 2 Complete. Ready for Phase 3 Inference Deployment.")

if __name__ == '__main__':
    main()
