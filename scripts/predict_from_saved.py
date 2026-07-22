"""
Prediction Script from Pre-Trained Models (Colab Workflow)

Usage:
python scripts/predict_from_saved.py --market madhur --folder backtest_winners

This script loads the pre-trained models from the Google Drive/Colab sync.
If using 'backtest_winners', it loads the intelligent weights from state.json
and applies confidence calibrations.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import json
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_1_1.features import load_raw_data, build_prediction_features, ALL_FEATURE_GROUPS, MARKET_CONFIG
from hybrid_1_1.models import MODEL_TYPES

def get_next_playing_date(last_date, market):
    cfg = MARKET_CONFIG[market]
    playing_days_per_week = cfg['playing_days_per_week']
    next_date = last_date + timedelta(days=1)
    if playing_days_per_week == 6:
        while next_date.weekday() == 6: next_date += timedelta(days=1)
    elif playing_days_per_week == 5:
        while next_date.weekday() >= 5: next_date += timedelta(days=1)
    return next_date

def format_prediction(probs, thresholds=None):
    top_3_idx = np.argsort(probs)[-3:][::-1]
    top_prob = probs[top_3_idx[0]]
    
    confidence = ""
    if thresholds:
        if top_prob >= thresholds.get('strong', 1.0): confidence = " [STRONG]"
        elif top_prob >= thresholds.get('good', 1.0): confidence = " [GOOD]"
        elif top_prob >= thresholds.get('marginal', 1.0): confidence = " [MARGINAL]"
        else: confidence = " [WEAK]"

    output = []
    for idx in top_3_idx:
        pct = probs[idx] * 100
        output.append(f"{idx} ({pct:.1f}%)")
    return " | ".join(output) + confidence

def run_prediction(market, folder):
    dir_path = os.path.join("trained_models", folder, market)
    if not os.path.exists(dir_path):
        print(f"ERROR: Cannot find trained models at {dir_path}")
        sys.exit(1)
        
    print(f"\n{'#'*70}")
    print(f"  PREDICTING TOMORROW FROM SAVED COLAB MODELS: {market.upper()} ({folder})")
    print(f"{'#'*70}")

    df = load_raw_data(market)
    last_date = pd.to_datetime(df['Date'].iloc[-1]).date()
    pred_date = get_next_playing_date(last_date, market)
    
    print(f"\nLast Draw: {last_date}  -->  Predicting For: {pred_date.strftime('%Y-%m-%d (%A)')}")
    
    # Load state if using intelligent backtest winners
    weights_m, weights_e = None, None
    thresholds_m, thresholds_e = None, None
    surviving_groups = ALL_FEATURE_GROUPS
    
    if folder == 'backtest_winners':
        state_path = os.path.join("hybrid_1_1", "state", f"{market}_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            weights_m = state.get('weights_m', {})
            weights_e = state.get('weights_e', {})
            thresholds_m = state.get('thresholds_m', {})
            thresholds_e = state.get('thresholds_e', {})
            surviving_groups = state.get('surviving_groups', ALL_FEATURE_GROUPS)
            print("Loaded intelligent weights & calibrations from state.json")
        else:
            print("WARNING: backtest_winners folder specified, but no state.json found. Defaulting to raw average.")

    print("Building features for tomorrow...")
    X_m_last, X_e_last, _ = build_prediction_features(df, surviving_groups)
    
    seq_m = df['Morning_number'].dropna().astype(int).values
    seq_e = df['Evening_number'].dropna().astype(int).values
    current_dow = pred_date.weekday()

    final_probs_m = np.zeros(10)
    final_probs_e = np.zeros(10)
    
    loaded_m = []
    loaded_e = []

    print("\n--- Model Predictions ---")
    
    # Identify which models to load based on weights (if present), else load all
    active_m_keys = set(weights_m.keys()) if weights_m else None
    active_e_keys = set(weights_e.keys()) if weights_e else None

    # We iterate over the directory to see what's available
    for name, ModelClass in MODEL_TYPES.items():
        if name == 'ag': continue
        
        # We need to handle model IDs like "3m_xgb"
        for window in ['1m', '2m', '3m', 'full']:
            model_id = f"{window}_{name}"
            
            # Morning
            if active_m_keys is None or model_id in active_m_keys:
                model_m = ModelClass()
                try:
                    model_m.load_models(dir_path, f"{model_id}_morning")
                    probs_m = model_m.predict_proba(X_m_last, last_digits=seq_m, current_dow=current_dow)
                    w = weights_m[model_id] if weights_m else 1.0
                    final_probs_m += probs_m * w
                    loaded_m.append(model_id)
                except FileNotFoundError:
                    pass
            
            # Evening
            if active_e_keys is None or model_id in active_e_keys:
                model_e = ModelClass()
                try:
                    model_e.load_models(dir_path, f"{model_id}_evening")
                    probs_e = model_e.predict_proba(X_e_last, last_digits=seq_e, current_dow=current_dow)
                    w = weights_e[model_id] if weights_e else 1.0
                    final_probs_e += probs_e * w
                    loaded_e.append(model_id)
                except FileNotFoundError:
                    pass

    if not loaded_m and not loaded_e:
        print("Error: No models were successfully loaded! Check if the folder exists and has files.")
        return

    # Normalize if not using state weights
    if not weights_m and sum(final_probs_m) > 0: final_probs_m /= sum(final_probs_m)
    if not weights_e and sum(final_probs_e) > 0: final_probs_e /= sum(final_probs_e)

    print(f"Loaded {len(loaded_m)} Morning Models")
    print(f"Loaded {len(loaded_e)} Evening Models")
    
    print(f"\n{'='*70}")
    print(f"MORNING PREDICTION: {format_prediction(final_probs_m, thresholds_m)}")
    print(f"EVENING PREDICTION: {format_prediction(final_probs_e, thresholds_e)}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', type=str, default='madhur', choices=['kalyan', 'mb', 'madhur'])
    parser.add_argument('--folder', type=str, default='backtest_winners')
    args = parser.parse_args()
    run_prediction(args.market, args.folder)
