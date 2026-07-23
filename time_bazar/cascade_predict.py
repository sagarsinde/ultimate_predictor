"""
Cascade Jodi Engine
Predicts Morning first, then INJECTS the predicted Morning digit into the Evening features
to generate a conditional (P(E|M)) prediction for Jodi.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import timedelta

# Add the market directory to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import load_raw_data, build_prediction_features, ALL_FEATURE_GROUPS, PLAYING_DAYS_PER_WEEK
from src.models import MODEL_TYPES

def get_next_playing_date(last_date):
    next_date = last_date + timedelta(days=1)
    if PLAYING_DAYS_PER_WEEK == 6:
        while next_date.weekday() == 6: next_date += timedelta(days=1)
    elif PLAYING_DAYS_PER_WEEK == 5:
        while next_date.weekday() >= 5: next_date += timedelta(days=1)
    return next_date

def run_cascade_prediction():
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.exists(dir_path):
        print(f"ERROR: Cannot find trained models at {dir_path}")
        sys.exit(1)
        
    print(f"\n{'='*70}")
    print(f"  CASCADE JODI ENGINE")
    print(f"{'='*70}")

    df = load_raw_data()
    last_date = pd.to_datetime(df['Date'].iloc[-1]).date()
    pred_date = get_next_playing_date(last_date)
    current_dow = pred_date.weekday()
    
    print(f"\nLast Draw: {last_date}  -->  Predicting For: {pred_date.strftime('%Y-%m-%d (%A)')}")
    
    state_path = os.path.join(dir_path, "state.json")
    if not os.path.exists(state_path):
        print("ERROR: state.json not found! Please run backtest first.")
        sys.exit(1)
        
    with open(state_path, 'r') as f:
        state = json.load(f)
        
    weights_m = state.get('weights_m', {})
    weights_e = state.get('weights_e', {})
    surviving_groups = state.get('surviving_groups', ALL_FEATURE_GROUPS)
    
    print("Building base features...")
    X_m_last_df, X_e_last_df, _ = build_prediction_features(df, surviving_groups)
    
    feat_cols_e = X_e_last_df.columns.tolist()
    X_m_last = X_m_last_df.values
    X_e_last = X_e_last_df.values
    
    seq_m = df['Morning_number'].dropna().astype(int).values
    seq_e = df['Evening_number'].dropna().astype(int).values
    
    # ---------------------------------------------------------
    # 1. MORNING PREDICTION
    # ---------------------------------------------------------
    print("\n  [1/3] Predicting Morning Draw...")
    ensemble_m_probs = np.zeros(10)
    total_weight_m = sum(weights_m.values())
    
    for name, ModelClass in MODEL_TYPES.items():
        if name == 'ag': continue
        for window in ['1m', '2m', '3m', 'full']:
            model_id = f"{window}_{name}"
            if model_id not in weights_m: continue
                
            model_m = ModelClass()
            try:
                model_m.load_models(dir_path, f"{model_id}_morning")
                probs_m = model_m.predict_proba(X_m_last, last_digits=seq_m, current_dow=current_dow)
                ensemble_m_probs += probs_m * weights_m[model_id]
            except Exception:
                pass

    if total_weight_m > 0:
        ensemble_m_probs /= total_weight_m
        
    m_ranking = np.argsort(ensemble_m_probs)[::-1]
    top_3_morning = m_ranking[:3]
    
    print(f"  Morning Top-3 Predicted Digits: {top_3_morning.tolist()}")
    for i, d in enumerate(top_3_morning, 1):
        print(f"    Rank {i}: {d} ({ensemble_m_probs[d]*100:.1f}%)")

    # ---------------------------------------------------------
    # 2. CASCADE INJECTION FOR EVENING
    # ---------------------------------------------------------
    print(f"\n  [2/3] Cascade Injection (Predicting Evening given Morning)...")
    
    try:
        m_inject_idx = feat_cols_e.index('Today_Morning')
    except ValueError:
        m_inject_idx = -1
        
    me_corr_indices = {}
    for d in range(10):
        try:
            me_corr_indices[d] = feat_cols_e.index(f'ME_corr_{d}')
        except ValueError:
            pass

    joint_counts = np.zeros((10, 10), dtype=float)
    morning_vals = df['Morning_number'].astype(int).values
    evening_vals = df['Evening_number'].astype(int).values
    for m_val, e_val in zip(morning_vals, evening_vals):
        joint_counts[m_val][e_val] += 1
        
    cascade_jodis = {}

    for m_cand in top_3_morning:
        m_prob = ensemble_m_probs[m_cand]
        ensemble_e_probs = np.zeros(10)
        total_weight_e = sum(weights_e.values())
        
        for name, ModelClass in MODEL_TYPES.items():
            if name == 'ag': continue
            for window in ['1m', '2m', '3m', 'full']:
                model_id = f"{window}_{name}"
                if model_id not in weights_e: continue
                
                # We need to alter X_e_last for this specific morning candidate
                X_injected = X_e_last.copy()
                
                if m_inject_idx != -1:
                    X_injected[0, m_inject_idx] = m_cand
                    
                if me_corr_indices:
                    row_total = joint_counts[m_cand].sum()
                    if row_total > 0:
                        probs = joint_counts[m_cand] / row_total
                    else:
                        probs = np.ones(10) / 10.0
                    for d in range(10):
                        idx = me_corr_indices.get(d)
                        if idx is not None:
                            X_injected[0, idx] = probs[d]
                            
                # Predict
                model_e = ModelClass()
                try:
                    model_e.load_models(dir_path, f"{model_id}_evening")
                    
                    # For sequential models, we pretend the morning digit just happened
                    probs_e = model_e.predict_proba(X_injected, last_digits=seq_e, current_dow=current_dow)
                    ensemble_e_probs += probs_e * weights_e[model_id]
                except Exception as e:
                    print(f"DEBUG: Exception on evening model {model_id}: {e}")

        if total_weight_e > 0:
            ensemble_e_probs /= total_weight_e
            
        top_e = np.argsort(ensemble_e_probs)[::-1][0]
        top_e_prob = ensemble_e_probs[top_e]
        
        jodi = f"{m_cand}{top_e}"
        cascade_prob = m_prob * top_e_prob
        cascade_jodis[jodi] = cascade_prob
        
        print(f"    If Morning is {m_cand} -> Evening predicts {top_e} (Cond Prob: {top_e_prob*100:.1f}%) -> Jodi {jodi}")

    # ---------------------------------------------------------
    # 3. FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n  [3/3] Final Cascade Jodi Predictions")
    print(f"  {'-'*40}")
    
    sorted_cascade = sorted(cascade_jodis.items(), key=lambda x: x[1], reverse=True)
    for rank, (jodi, prob) in enumerate(sorted_cascade, 1):
        print(f"    Rank {rank}: Jodi {jodi} (Combined Prob: {prob*100:.2f}%)")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    run_cascade_prediction()
