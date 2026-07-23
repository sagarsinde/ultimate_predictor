import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from hybrid_1_1.data_loader import DataLoader
from hybrid_1_1.features import FeatureEngineer
from hybrid_1_1.autogluon_model import AutoGluonModel

def train_autogluon_standalone(market, time_limit=3600):
    print(f"==================================================")
    print(f"  TRAINING AUTOGLUON STANDALONE FOR {market.upper()}")
    print(f"==================================================")
    
    # 1. Load Data
    loader = DataLoader(market=market)
    df = loader.load_data()
    
    if df is None or len(df) == 0:
        print(f"Failed to load data for {market}")
        return
        
    # 2. Build Features
    fe = FeatureEngineer()
    df_features = fe.build_features(df)
    
    print(f"Total rows: {len(df_features)}")
    
    # 3. Create directories
    out_dir = os.path.join("trained_models", "autogluon_winners")
    os.makedirs(out_dir, exist_ok=True)
    
    # 4. Train Morning Model
    print("\n--- Training Morning AutoGluon Model ---")
    y_m = df_features['Morning_number']
    
    # Drop future-leaking columns
    cols_to_drop = [
        'Morning_number', 'Evening_number', 
        'Morning_card1', 'Morning_card2', 'Morning_card3',
        'Evening_number1', 'Evening_number2', 'Evening_number3',
        'Date'
    ]
    
    X_m = df_features.drop(columns=[c for c in cols_to_drop if c in df_features.columns])
    
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
    y_e = df_features['Evening_number']
    
    # Evening uses the same features as morning to predict the evening draw independently
    X_e = X_m.copy()
    
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
