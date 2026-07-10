import numpy as np
import pandas as pd
from datetime import timedelta

from hybrid_1_1.features import (
    load_raw_data, build_features, slice_window,
    get_window_size, MARKET_CONFIG,
)
from hybrid_1_1.models import MODEL_TYPES, FEATURE_MODELS
from hybrid_1_1.validator import load_state, _train_single_model, _predict_single
from hybrid_1_1.predict import get_next_playing_date

def run_cascade_prediction(market='kalyan'):
    state = load_state(market)
    if not state:
        print(f"ERROR: No saved state for {market}.")
        return

    weights_m = state.get('weights_m', {})
    weights_e = state.get('weights_e', {})
    surviving_groups = state['surviving_groups']
    
    active_models = set(weights_m.keys()) | set(weights_e.keys())
    
    df = load_raw_data(market)
    last_date = pd.to_datetime(df['Date'].iloc[-1]).date()
    pred_date = get_next_playing_date(last_date, market)
    pred_date_str = pred_date.strftime('%Y-%m-%d (%A)')

    print(f"\n{'═'*70}")
    print(f"  CASCADE ENGINE: {market.upper()} PREDICTION FOR {pred_date_str}")
    print(f"{'═'*70}")

    # 1. Train all models and get Morning Predictions
    ensemble_m_probs = np.zeros(10)
    total_weight_m = sum(weights_m.values())
    
    # Store trained models and base features so we can reuse them for Evening Cascade
    trained_models = {}
    base_X_pred = None
    feature_cols = None
    
    last_m_vals = df['Morning_number'].iloc[-2:].astype(int).tolist()
    if len(last_m_vals) == 1: last_m_vals = [last_m_vals[0], last_m_vals[0]]
    elif len(last_m_vals) == 0: last_m_vals = [0, 0]
    
    last_e_vals = df['Evening_number'].iloc[-2:].astype(int).tolist()
    if len(last_e_vals) == 1: last_e_vals = [last_e_vals[0], last_e_vals[0]]
    elif len(last_e_vals) == 0: last_e_vals = [0, 0]
    
    current_dow = pred_date.weekday()

    print("  [1/3] Training Models and Predicting Morning Draw...")
    for model_id in sorted(active_models):
        weight_m = weights_m.get(model_id, 0.0)
        parts = model_id.split('_', 1)
        window_label, model_type = parts[0], parts[1]

        model_m_obj, model_e_obj, feat_cols = _train_single_model(
            model_type, window_label, df, market, surviving_groups
        )
        
        if model_m_obj is None:
            continue
            
        window_draws = get_window_size(market, window_label)
        from hybrid_1_1.features import build_prediction_features
        X_pred_m_df, X_pred_e_df, _ = build_prediction_features(df, surviving_groups)
        
        feat_cols_m = X_pred_m_df.columns.tolist()
        feat_cols_e = X_pred_e_df.columns.tolist()
        X_pred_m = X_pred_m_df.values
        X_pred_e = X_pred_e_df.values
        
        if base_X_pred is None:
            base_X_pred = X_pred_e.copy()
            feature_cols = feat_cols_e

        m_probs, _ = _predict_single(
            model_m_obj, model_e_obj, model_type, X_pred_m, X_pred_e, last_m_vals, last_e_vals, current_dow
        )
        
        if weight_m > 0:
            ensemble_m_probs += weight_m * m_probs
            
        trained_models[model_id] = {
            'e_obj': model_e_obj,
            'type': model_type,
            'X_pred_e': X_pred_e,
            'weight_e': weights_e.get(model_id, 0.0)
        }

    if total_weight_m > 0:
        ensemble_m_probs /= total_weight_m

    m_ranking = np.argsort(ensemble_m_probs)[::-1]
    top_3_morning = m_ranking[:3]

    print(f"  Morning Top-3 Predicted Digits: {top_3_morning.tolist()}")
    for i, d in enumerate(top_3_morning, 1):
        print(f"    Rank {i}: {d} ({ensemble_m_probs[d]*100:.1f}%)")

    # 2. Inject each Morning Digit and predict Evening
    print(f"\n  [2/3] Cascade Injection (Predicting Evening given Morning Result)...")
    
    try:
        m_inject_idx = feature_cols.index('Today_Morning')
    except ValueError:
        m_inject_idx = -1
        
    me_corr_indices = {}
    for d in range(10):
        try:
            me_corr_indices[d] = feature_cols.index(f'ME_corr_{d}')
        except ValueError:
            pass

    # We need joint counts to calculate new ME_corr
    joint_counts = np.zeros((10, 10), dtype=float)
    morning_vals = df['Morning_number'].astype(int).values
    evening_vals = df['Evening_number'].astype(int).values
    for m_val, e_val in zip(morning_vals, evening_vals):
        joint_counts[m_val][e_val] += 1
    
    total_weight_e = sum(weights_e.values())
    cascade_jodis = {}

    for m_cand in top_3_morning:
        m_prob = ensemble_m_probs[m_cand]
        
        # Ensemble evening probs given this specific morning digit
        ensemble_e_probs = np.zeros(10)
        
        for model_id, m_data in trained_models.items():
            weight_e = m_data['weight_e']
            if weight_e == 0:
                continue
                
            model_e_obj = m_data['e_obj']
            model_type = m_data['type']
            
            # Create a localized copy of the feature vector to inject into
            X_injected = m_data['X_pred_e'].copy()
            
            # INJECTION: Replace Today_Morning with the predicted Morning candidate
            if m_inject_idx != -1:
                X_injected[0, m_inject_idx] = m_cand
                
            # INJECTION: Replace ME_corr with conditional probability given m_cand
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
            
            # Construct a temporary last_m_vals to feed to Markov models
            fake_last_m_vals = [last_m_vals[-1], m_cand]
            
            # Predict evening using the injected features and fake last_m
            _, e_probs = _predict_single(
                None, model_e_obj, model_type, None, X_injected, fake_last_m_vals, last_e_vals, current_dow
            )
            
            ensemble_e_probs += weight_e * e_probs
            
        if total_weight_e > 0:
            ensemble_e_probs /= total_weight_e
            
        top_e = np.argsort(ensemble_e_probs)[::-1][0] # Just grab the Top-1 Evening for this specific cascade path
        top_e_prob = ensemble_e_probs[top_e]
        
        jodi = f"{m_cand}{top_e}"
        # Combined probability = P(Morning) * P(Evening | Morning)
        cascade_prob = m_prob * top_e_prob
        cascade_jodis[jodi] = cascade_prob
        
        print(f"    If Morning is {m_cand} -> Cascade predicts Evening is {top_e} (Cond Prob: {top_e_prob*100:.1f}%) -> Jodi {jodi}")

    print("\n  [3/3] Final Cascade Jodi Predictions")
    print(f"  {'─'*40}")
    
    sorted_cascade = sorted(cascade_jodis.items(), key=lambda x: x[1], reverse=True)
    for rank, (jodi, prob) in enumerate(sorted_cascade, 1):
        print(f"    Rank {rank}: Jodi {jodi} (Combined Prob: {prob*100:.2f}%)")
    print(f"{'═'*70}\n")

if __name__ == '__main__':
    import sys
    market = sys.argv[1] if len(sys.argv) > 1 else 'kalyan'
    run_cascade_prediction(market)
