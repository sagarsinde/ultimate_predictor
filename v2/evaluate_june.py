"""
v2/evaluate_june.py — Compare predictions vs reality for June 2026

Trains the ensemble using data up to the day before, predicts the current day,
and compares the prediction to the ACTUAL result for every day in June.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.features import load_raw_data, build_features, slice_window, get_window_size
from v2.validator import load_state, _train_single_model, _predict_single

def evaluate_month(market, target_month='2026-06'):
    state = load_state(market)
    if state is None:
        print(f"No saved state for {market}. Run backtest first.")
        return

    weights = state['weights']
    surviving_groups = state['surviving_groups']
    
    df = load_raw_data(market)
    
    # Get all indices for the target month
    df['Date'] = pd.to_datetime(df['Date'])
    june_mask = df['Date'].dt.to_period('M') == target_month
    june_indices = df.index[june_mask].tolist()
    
    if not june_indices:
        print(f"No data found for {target_month}")
        return
        
    print(f"\n{'='*70}")
    print(f"  EVALUATING ENSEMBLE VS REALITY: {market.upper()} ({target_month})")
    print(f"{'='*70}")
    print(f"  {'Date':<12} | {'Actual M':<8} | {'Predicted Top-3 (Morning)':<30} | {'Hit?':<5}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*30}-+-{'-'*5}")
    
    hits_top1 = 0
    hits_top3 = 0
    total = 0
    
    for pred_idx in june_indices:
        pred_date = df.iloc[pred_idx]['Date'].strftime('%Y-%m-%d')
        actual_m = int(df.iloc[pred_idx]['Morning_number'])
        actual_e = int(df.iloc[pred_idx]['Evening_number'])
        
        # Context is everything UP TO the day before
        context_df = df.iloc[:pred_idx].copy()
        
        ensemble_m_probs = np.zeros(10)
        total_weight = 0.0
        
        # The user's requested short-term models
        custom_models = [
            '3m_rf', '3m_xgb', '2m_rf', '2m_xgb', 
            '1m_freq', '1m_markov', '1m_rf'
        ]
        
        # Override weights to ONLY use these custom models equally
        if custom_models:
            weights = {m: 1.0/len(custom_models) for m in custom_models}

        for model_id, weight in weights.items():
            parts = model_id.split('_', 1)
            window_label = parts[0]
            model_type = parts[1]
            
            model_m, model_e, _ = _train_single_model(
                model_type, window_label, context_df, market, surviving_groups
            )
            
            if model_m is None: continue
            
            window_draws = get_window_size(market, window_label)
            pred_df = slice_window(context_df, window_draws)
            feat_df, _, _, _ = build_features(pred_df, surviving_groups)
            
            if len(feat_df) == 0: continue
            
            last_row = feat_df.iloc[[-1]]
            feat_only = [c for c in last_row.columns if c != '_date']
            X_pred = last_row[feat_only].values
            
            last_m = int(context_df.iloc[-1]['Morning_number'])
            last_e = int(context_df.iloc[-1]['Evening_number'])
            
            m_probs, _ = _predict_single(model_m, model_e, model_type, X_pred, last_m, last_e)
            
            ensemble_m_probs += weight * m_probs
            total_weight += weight
            
        if total_weight > 0:
            ensemble_m_probs /= total_weight
            
        m_ranking = np.argsort(ensemble_m_probs)[::-1]
        top3_m = m_ranking[:3].tolist()
        
        # Check hits
        total += 1
        is_top1 = (actual_m == top3_m[0])
        is_top3 = (actual_m in top3_m)
        
        if is_top1:
            hits_top1 += 1
            hit_marker = "⭐⭐⭐ TOP 1!"
        elif is_top3:
            hits_top3 += 1
            hit_marker = "✅ Top 3"
        else:
            hit_marker = "❌ Miss"
            
        top3_str = f"{top3_m[0]}, {top3_m[1]}, {top3_m[2]}"
        print(f"  {pred_date:<12} | {actual_m:<8} | {top3_str:<30} | {hit_marker}")

    print(f"\n{'='*70}")
    print(f"  RESULTS FOR {target_month}:")
    print(f"  Top-1 Hits: {hits_top1}/{total} ({(hits_top1/total)*100:.1f}%)")
    print(f"  Top-3 Hits: {(hits_top1 + hits_top3)}/{total} ({((hits_top1 + hits_top3)/total)*100:.1f}%)")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    evaluate_month('kalyan', '2026-06')
