"""
Train ONLY the heavy AutoGluon model on the full historical data.
This script is designed to run overnight on Google Colab.
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_1_1.features import load_raw_data, build_features, ALL_FEATURE_GROUPS

try:
    from hybrid_1_1.autogluon_model import AutoGluonModel
except ImportError:
    print("AutoGluon is not installed. Please install it on Colab using: !pip install autogluon")
    sys.exit(1)

def train_autogluon_heavy(market):
    print(f"\n{'='*60}")
    print(f"--- HEAVY AUTOGLUON TRAINING: {market.upper()} ---")
    print(f"{'='*60}")
    
    df = load_raw_data(market)
    print(f"Loaded {len(df)} draws (Full History). Building features...")
    
    X_m, X_e, y_m, y_e, _ = build_features(df, ALL_FEATURE_GROUPS)
    
    if '_date' in X_m.columns: X_m = X_m.drop(columns=['_date'])
    if '_date' in X_e.columns: X_e = X_e.drop(columns=['_date'])
    
    seq_m = y_m.values
    seq_e = y_e.values
    
    dir_path = os.path.join("trained_models", "autogluon_heavy", market)
    os.makedirs(dir_path, exist_ok=True)
    
    print(f"  -> Training AutoGluon Morning Model (This will take hours)...")
    model_m = AutoGluonModel()
    model_m.fit(X_m, y_m, sequence=seq_m)
    # AutoGluon internally saves its state to an AutogluonModels folder during training,
    # but we will try to unify it.
    if hasattr(model_m, 'save_models'):
        model_m.save_models(dir_path, "ag_morning")
    
    print(f"  -> Training AutoGluon Evening Model (This will take hours)...")
    model_e = AutoGluonModel()
    model_e.fit(X_e, y_e, sequence=seq_e)
    if hasattr(model_e, 'save_models'):
        model_e.save_models(dir_path, "ag_evening")
        
    print(f"\nSuccess! AutoGluon Models completed and saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', type=str, default='madhur', choices=['kalyan', 'mb', 'madhur'])
    args = parser.parse_args()
    train_autogluon_heavy(args.market)
