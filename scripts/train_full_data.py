"""
Train all standardized fast models on the FULL historical data.
Saves models to trained_models/full_data/<market>/
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_1_1.features import load_raw_data, slice_window, get_window_size, build_features, ALL_FEATURE_GROUPS
from hybrid_1_1.models import MODEL_TYPES, FEATURE_MODELS, SEQUENCE_MODELS

def train_full_data(market):
    print(f"\n--- Training Full Data Models for {market.upper()} ---")
    
    df = load_raw_data(market)
    
    print(f"Loaded {len(df)} draws (Full History). Building features...")
    X_m, X_e, y_m, y_e, _ = build_features(df, ALL_FEATURE_GROUPS)
    
    if '_date' in X_m.columns: X_m = X_m.drop(columns=['_date'])
    if '_date' in X_e.columns: X_e = X_e.drop(columns=['_date'])
    
    seq_m = y_m.values
    seq_e = y_e.values
    
    dir_path = os.path.join("trained_models", "full_data", market)
    os.makedirs(dir_path, exist_ok=True)
    
    for name, ModelClass in MODEL_TYPES.items():
        if name == 'ag': continue
        
        print(f"  -> Training {name.upper()}...")
        model_m = ModelClass()
        model_e = ModelClass()
        
        if name in FEATURE_MODELS:
            model_m.fit(X_m, y_m, sequence=seq_m)
            model_e.fit(X_e, y_e, sequence=seq_e)
        elif name in SEQUENCE_MODELS:
            model_m.fit(None, None, sequence=seq_m)
            model_e.fit(None, None, sequence=seq_e)
            
        model_m.save_models(dir_path, f"{name}_morning")
        model_e.save_models(dir_path, f"{name}_evening")
        
    print(f"Success! All fast models saved to {dir_path}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', type=str, default='madhur', choices=['kalyan', 'mb', 'madhur'])
    args = parser.parse_args()
    train_full_data(args.market)
