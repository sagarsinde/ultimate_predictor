import sys
import numpy as np
from v4_jodi.features import load_raw_data, build_features, slice_window, get_window_size
from v4_jodi.validator import load_state
from v4_jodi.models import MODEL_TYPES

def predict_tomorrow(market):
    state = load_state(market)
    if not state:
        print(f"ERROR: No learned state for {market}. Run backtest first.")
        return

    weights = state['weights']
    active_groups = state['surviving_groups']
    df = load_raw_data(market)
    
    print(f"\n{'='*60}")
    print(f"  V4 JODI PREDICTION: {market.upper()}")
    print(f"  Using {len(weights)} models (Active features: {active_groups})")
    print(f"{'='*60}\n")

    ensemble_probs = np.zeros(100)

    for model_id, weight in weights.items():
        wl, mt = model_id.split('_')
        window_draws = get_window_size(market, wl)
        sliced_df = slice_window(df, window_draws)
        
        feature_df, y, _ = build_features(sliced_df, active_groups)
        feature_cols = [c for c in feature_df.columns if c != '_date']
        X_train = feature_df[feature_cols].values
        
        model_cls = MODEL_TYPES[mt]
        model = model_cls()
        model.fit(X_train, y.values)
        
        last_row_feats = feature_df.iloc[[-1]]
        X_pred = last_row_feats[feature_cols].values
        
        probs = model.predict_proba(X_pred)
        ensemble_probs += probs * weight
        
        top1 = np.argmax(probs)
        print(f"  {model_id:<12} (w={weight:.4f})  -> Top 1: {top1:02d}")

    print(f"\n{'-'*60}")
    print(f"  ENSEMBLE FINAL JODI PREDICTION (TOP 10)")
    print(f"{'-'*60}")
    
    sorted_indices = np.argsort(ensemble_probs)[::-1]
    for rank in range(1, 11):
        idx = sorted_indices[rank-1]
        print(f"  Rank {rank:<2}: {idx:02d}  ({ensemble_probs[idx]*100:.2f}%)")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    market = sys.argv[1].lower() if len(sys.argv) > 1 else 'kalyan'
    predict_tomorrow(market)
