import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from hybrid_1_1.features import load_raw_data, build_features, ALL_FEATURE_GROUPS
from hybrid_1_1.autogluon_model import AutoGluonModel

def train_autogluon_standalone(market, time_limit=3600):
    print(f"==================================================")
    print(f"  TRAINING AUTOGLUON STANDALONE FOR {market.upper()}")
    print(f"==================================================")
    
    # 1. Load Data
    df = load_raw_data(market)
    
    if df is None or len(df) == 0:
        print(f"Failed to load data for {market}")
        return
        
    # 2. Build Features
    X_m, X_e, y_m, y_e, _ = build_features(df, ALL_FEATURE_GROUPS)
    
    if '_date' in X_m.columns: X_m = X_m.drop(columns=['_date'])
    if '_date' in X_e.columns: X_e = X_e.drop(columns=['_date'])
    
    print(f"Total rows: {len(X_m)}")
    
    # 3. Create directories
    out_dir = os.path.join("trained_models", "autogluon_winners")
    os.makedirs(out_dir, exist_ok=True)
    
    # 4. Train Morning Model
    print("\n--- Training Morning AutoGluon Model ---")
    
    m_model = AutoGluonModel()
    m_path = os.path.join(out_dir, f"{market}_ag_morning_temp")
    
    m_model.fit(
        X=X_m, 
        y=y_m, 
        time_limit=time_limit, 
        presets='best_quality', 
        path=m_path
    )
    
    if m_model.predictor is not None:
        m_model.save_models(out_dir, f"{market}_ag_morning")
        print("Morning AutoGluon model saved successfully.")
    
    # 5. Train Evening Model
    print("\n--- Training Evening AutoGluon Model ---")
    
    e_model = AutoGluonModel()
    e_path = os.path.join(out_dir, f"{market}_ag_evening_temp")
    
    e_model.fit(
        X=X_e, 
        y=y_e, 
        time_limit=time_limit, 
        presets='best_quality', 
        path=e_path
    )
    
    if e_model.predictor is not None:
        e_model.save_models(out_dir, f"{market}_ag_evening")
        print("Evening AutoGluon model saved successfully.")
        
    print("\nTraining Complete! You can now use predict_from_saved.py to see AutoGluon predictions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("market", type=str)
    parser.add_argument("--time_limit", type=int, default=3600, help="Time limit in seconds per model")
    args = parser.parse_args()
    
    train_autogluon_standalone(args.market, args.time_limit)
