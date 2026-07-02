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

    surviving_groups = state['surviving_groups']
    df = load_raw_data(market)
    
    # Get all indices for the target month
    df['Date'] = pd.to_datetime(df['Date'])
    june_mask = df['Date'].dt.to_period('M') == target_month
    june_indices = df.index[june_mask].tolist()
    
    if not june_indices:
        print(f"No data found for {target_month}")
        return
        
    print(f"\n{'='*100}")
    print(f"  EVALUATING: {market.upper()} ({target_month}) — ONLY full_freq (Morning, Evening, Jodi)")
    print(f"{'='*100}")
    print(f"  {'Date':<12} | {'Actual M':<8} | {'Pred M (Top3)':<20} | {'Actual E':<8} | {'Pred E (Top3)':<20} | {'Actual Jodi':<12} | {'Pred Jodi (Top4)':<35}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*20}-+-{'-'*8}-+-{'-'*20}-+-{'-'*12}-+-{'-'*35}")
    
    total = 0
    m_top1_hits = 0; m_top3_hits = 0
    e_top1_hits = 0; e_top3_hits = 0
    jodi_hits = 0
    
    for pred_idx in june_indices:
        pred_date = df.iloc[pred_idx]['Date'].strftime('%Y-%m-%d')
        actual_m = int(df.iloc[pred_idx]['Morning_number'])
        actual_e = int(df.iloc[pred_idx]['Evening_number'])
        actual_jodi = f"{actual_m}{actual_e}"
        
        # Context is everything UP TO the day before
        context_df = df.iloc[:pred_idx].copy()
        
        # We only want to use 3m_freq
        model_id = '3m_freq'
        window_label, model_type = '3m', 'freq'
        
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
        
        m_probs, e_probs = _predict_single(model_m, model_e, model_type, X_pred, last_m, last_e)
        
        # Morning Rankings
        m_ranking = np.argsort(m_probs)[::-1]
        top3_m = m_ranking[:3].tolist()
        
        # Evening Rankings
        e_ranking = np.argsort(e_probs)[::-1]
        top3_e = e_ranking[:3].tolist()
        
        # Jodi Rankings
        jodi_probs = np.outer(m_probs, e_probs)
        flat_jodi_probs = jodi_probs.flatten()
        jodi_ranking = np.argsort(flat_jodi_probs)[::-1]
        
        top4_jodis = []
        for rank in jodi_ranking[:4]:
            m_digit = rank // 10
            e_digit = rank % 10
            top4_jodis.append(f"{m_digit}{e_digit}")
            
        # Checks
        total += 1
        
        m_hit_str = ""
        if actual_m == top3_m[0]: m_top1_hits += 1; m_hit_str = "⭐"
        elif actual_m in top3_m: m_top3_hits += 1; m_hit_str = "✅"
        else: m_hit_str = "❌"
            
        e_hit_str = ""
        if actual_e == top3_e[0]: e_top1_hits += 1; e_hit_str = "⭐"
        elif actual_e in top3_e: e_top3_hits += 1; e_hit_str = "✅"
        else: e_hit_str = "❌"
            
        jodi_hit_str = ""
        if actual_jodi in top4_jodis:
            jodi_hits += 1
            jodi_hit_str = "⭐⭐⭐ JODI HIT!"
        else:
            jodi_hit_str = "❌"
            
        str_m = f"{top3_m[0]}, {top3_m[1]}, {top3_m[2]} {m_hit_str}"
        str_e = f"{top3_e[0]}, {top3_e[1]}, {top3_e[2]} {e_hit_str}"
        str_j = f"{', '.join(top4_jodis)} {jodi_hit_str}"
        
        print(f"  {pred_date:<12} | {actual_m:<8} | {str_m:<20} | {actual_e:<8} | {str_e:<20} | {actual_jodi:<12} | {str_j:<35}")

    print(f"\n{'='*100}")
    print(f"  FINAL RESULTS FOR {target_month} (using ONLY full_freq):")
    print(f"  Morning Top-1: {m_top1_hits}/{total} ({(m_top1_hits/total)*100:.1f}%)")
    print(f"  Morning Top-3: {(m_top1_hits + m_top3_hits)}/{total} ({((m_top1_hits + m_top3_hits)/total)*100:.1f}%)")
    print(f"  Evening Top-1: {e_top1_hits}/{total} ({(e_top1_hits/total)*100:.1f}%)")
    print(f"  Evening Top-3: {(e_top1_hits + e_top3_hits)}/{total} ({((e_top1_hits + e_top3_hits)/total)*100:.1f}%)")
    print(f"  Jodi Top-4:    {jodi_hits}/{total} ({(jodi_hits/total)*100:.1f}%)")
    print(f"{'='*100}\n")

if __name__ == '__main__':
    evaluate_month('kalyan', '2026-06')
