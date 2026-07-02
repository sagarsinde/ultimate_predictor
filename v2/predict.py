"""
v2/predict.py — Weighted Probability Ensemble + Calibrated Confidence

Loads surviving models + learned weights from validator output.
Trains only the surviving models on current data.
Combines weighted probabilities.
Applies evidence-based confidence labels.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from v2.features import (
    load_raw_data, build_features, slice_window,
    get_window_size, MARKET_CONFIG,
)
from v2.models import MODEL_TYPES, FEATURE_MODELS
from v2.validator import load_state, _train_single_model, _predict_single


def get_next_playing_date(last_date, market):
    """Calculate next playing day given market schedule."""
    cfg = MARKET_CONFIG[market]
    playing_days_per_week = cfg['playing_days_per_week']

    next_date = last_date + timedelta(days=1)
    if playing_days_per_week == 6:
        # Mon-Sat: skip Sunday (6)
        while next_date.weekday() == 6:
            next_date += timedelta(days=1)
    elif playing_days_per_week == 5:
        # Mon-Fri: skip Sat (5) and Sun (6)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

    return next_date


def predict_tomorrow(market, verbose=True):
    """
    Main prediction function.

    1. Load state (weights, surviving groups, thresholds, calibrators)
    2. Train surviving models on current data
    3. Get weighted ensemble probabilities
    4. Apply confidence calibration
    5. Print formatted output
    """
    state = load_state(market)
    if state is None:
        print(f"ERROR: No saved state for {market}. Run 'python v2/run_backtest.py {market}' first!")
        return

    weights = state['weights']
    surviving_groups = state['surviving_groups']
    thresholds_m = state.get('thresholds_m', {})
    thresholds_e = state.get('thresholds_e', {})
    calibrators = state.get('calibrators', {})

    if verbose:
        print(f"\n  Loading state: {len(weights)} surviving models, "
              f"{len(surviving_groups)} feature groups")

    # Load current data
    df = load_raw_data(market)

    # Determine prediction date
    last_date = pd.to_datetime(df['Date'].iloc[-1]).date()
    pred_date = get_next_playing_date(last_date, market)
    pred_date_str = pred_date.strftime('%Y-%m-%d (%A)')

    # Train each surviving model on current data
    ensemble_m_probs = np.zeros(10)
    ensemble_e_probs = np.zeros(10)
    model_details = []

    total_weight_applied = 0.0

    for model_id, weight in sorted(weights.items(), key=lambda x: -x[1]):
        # Parse model_id: "3m_xgb" -> window_label="3m", model_type="xgb"
        parts = model_id.split('_', 1)
        window_label = parts[0]
        model_type = parts[1]

        model_m, model_e, feat_cols = _train_single_model(
            model_type, window_label, df, market, surviving_groups
        )

        if model_m is None:
            continue

        # Build prediction features from full current data
        window_draws = get_window_size(market, window_label)
        pred_df = slice_window(df, window_draws)
        feat_df, _, _, _ = build_features(pred_df, surviving_groups)

        if len(feat_df) == 0:
            continue

        last_row = feat_df.iloc[[-1]]
        feat_only = [c for c in last_row.columns if c != '_date']
        X_pred = last_row[feat_only].values

        last_m = int(df.iloc[-1]['Morning_number'])
        last_e = int(df.iloc[-1]['Evening_number'])

        m_probs, e_probs = _predict_single(
            model_m, model_e, model_type, X_pred, last_m, last_e
        )

        # Weighted accumulation
        ensemble_m_probs += weight * m_probs
        ensemble_e_probs += weight * e_probs
        total_weight_applied += weight

        # Track per-model details
        model_details.append({
            'model_id': model_id,
            'weight': weight,
            'top_m': int(np.argmax(m_probs)),
            'top_m_prob': float(np.max(m_probs)),
            'top_e': int(np.argmax(e_probs)),
            'top_e_prob': float(np.max(e_probs)),
        })

    # Normalize (should already sum to ~1.0 but just in case)
    if total_weight_applied > 0:
        ensemble_m_probs /= total_weight_applied
        ensemble_e_probs /= total_weight_applied

    # Rank digits
    m_ranking = np.argsort(ensemble_m_probs)[::-1]
    e_ranking = np.argsort(ensemble_e_probs)[::-1]

    # Calibrate confidence
    top_m_prob = ensemble_m_probs[m_ranking[0]]
    top_e_prob = ensemble_e_probs[e_ranking[0]]

    m_signal = _get_confidence_label(top_m_prob, thresholds_m, calibrators.get('m'))
    e_signal = _get_confidence_label(top_e_prob, thresholds_e, calibrators.get('e'))

    # Compute historical hit rates if calibrator exists
    m_hist_rate = _get_historical_rate(top_m_prob, calibrators.get('m'))
    e_hist_rate = _get_historical_rate(top_e_prob, calibrators.get('e'))

    # Jodi predictions
    jodi_probs = {}
    for mm in range(10):
        for ee in range(10):
            jodi_probs[f"{mm}{ee}"] = ensemble_m_probs[mm] * ensemble_e_probs[ee]
    top4_jodis = sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]

    # ---- Print formatted output ----
    if verbose:
        _print_prediction(
            market, pred_date_str,
            ensemble_m_probs, ensemble_e_probs,
            m_ranking, e_ranking,
            m_signal, e_signal,
            m_hist_rate, e_hist_rate,
            top4_jodis, model_details,
            state,
        )

    return {
        'date': pred_date_str,
        'morning_probs': ensemble_m_probs,
        'evening_probs': ensemble_e_probs,
        'morning_signal': m_signal,
        'evening_signal': e_signal,
    }


def _get_confidence_label(top_prob, thresholds, calibrator):
    """Map predicted probability to evidence-based confidence label."""
    if not thresholds:
        # No calibration data — use raw probability heuristic
        if top_prob >= 0.18:
            return '🔥 STRONG'
        elif top_prob >= 0.14:
            return '✅ GOOD'
        elif top_prob >= 0.11:
            return '⚠️ MARGINAL'
        else:
            return '❌ SKIP'

    if top_prob >= thresholds.get('strong', 1.0):
        return '🔥 STRONG'
    elif top_prob >= thresholds.get('good', 1.0):
        return '✅ GOOD'
    elif top_prob >= thresholds.get('marginal', 1.0):
        return '⚠️ MARGINAL'
    else:
        return '❌ SKIP'


def _get_historical_rate(top_prob, calibrator):
    """Get calibrated historical hit rate for a given predicted probability."""
    if calibrator is None:
        return None
    try:
        return float(calibrator.predict(np.array([top_prob]))[0])
    except Exception:
        return None


def _print_prediction(market, pred_date_str, m_probs, e_probs,
                      m_ranking, e_ranking, m_signal, e_signal,
                      m_hist_rate, e_hist_rate,
                      top4_jodis, model_details, state):
    """Formatted prediction output."""

    print(f"\n{'═'*70}")
    print(f"  {market.upper()} PREDICTION: {pred_date_str}")
    print(f"{'═'*70}")

    # Morning
    print(f"\n  MORNING (Open):  [{m_signal}]")
    print(f"  {'─'*60}")
    print(f"  {'Rank':<6} {'Digit':<7} {'Ensemble Prob':<15} {'Hist. Hit Rate':<16}")
    print(f"  {'─'*6} {'─'*7} {'─'*15} {'─'*16}")
    for rank, d in enumerate(m_ranking[:5], 1):
        prob_str = f"{m_probs[d]*100:.1f}%"
        hist_str = f"{m_hist_rate*100:.1f}%" if m_hist_rate and rank == 1 else "—"
        print(f"  {rank:<6} {d:<7} {prob_str:<15} {hist_str:<16}")

    # Evening
    print(f"\n  EVENING (Close):  [{e_signal}]")
    print(f"  {'─'*60}")
    print(f"  {'Rank':<6} {'Digit':<7} {'Ensemble Prob':<15} {'Hist. Hit Rate':<16}")
    print(f"  {'─'*6} {'─'*7} {'─'*15} {'─'*16}")
    for rank, d in enumerate(e_ranking[:5], 1):
        prob_str = f"{e_probs[d]*100:.1f}%"
        hist_str = f"{e_hist_rate*100:.1f}%" if e_hist_rate and rank == 1 else "—"
        print(f"  {rank:<6} {d:<7} {prob_str:<15} {hist_str:<16}")

    # Jodi
    print(f"\n  TOP 4 JODI:")
    print(f"  {'─'*40}")
    for jodi, prob in top4_jodis:
        print(f"    Jodi {jodi}: {prob*100:.2f}%")

    # Model breakdown
    print(f"\n  MODEL BREAKDOWN:")
    print(f"  {'─'*60}")
    print(f"  {'Model':<15} {'Weight':<8} {'Top-M':<7} {'Prob-M':<8} {'Top-E':<7} {'Prob-E':<8}")
    print(f"  {'─'*15} {'─'*8} {'─'*7} {'─'*8} {'─'*7} {'─'*8}")
    for md in model_details:
        print(f"  {md['model_id']:<15} {md['weight']:<8.3f} "
              f"{md['top_m']:<7} {md['top_m_prob']*100:<7.1f}% "
              f"{md['top_e']:<7} {md['top_e_prob']*100:<7.1f}%")

    # Validation summary
    metrics = state.get('model_metrics', {})
    weights = state.get('weights', {})
    total_models = len(MODEL_TYPES) * 4  # 4 model types × 4 windows
    surviving = len(weights)
    feature_groups = state.get('surviving_groups', [])

    print(f"\n  VALIDATION SUMMARY:")
    print(f"  {'─'*60}")
    print(f"  Models active: {surviving} of {total_models} ({total_models - surviving} pruned)")
    print(f"  Features active: {len(feature_groups)} of 7 groups")
    print(f"  Active groups: {', '.join(feature_groups)}")
    print(f"  Last validated: {state.get('timestamp', 'unknown')}")
    print(f"{'═'*70}\n")
