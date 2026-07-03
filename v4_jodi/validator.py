import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from v4_jodi.features import load_raw_data, build_features, slice_window, get_window_size, ALL_FEATURE_GROUPS
from v4_jodi.models import MODEL_TYPES

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')

def _ensure_state_dir():
    os.makedirs(STATE_DIR, exist_ok=True)

def _get_validation_periods(df, num_periods=4, pred_days=7):
    df_dates = pd.to_datetime(df['Date'])
    months = sorted(df_dates.dt.to_period('M').unique())
    if len(months) < 2:
        raise ValueError("Not enough data")
    usable_months = months[-num_periods - 1:]
    periods = []
    for i in range(len(usable_months) - 1):
        train_month = usable_months[i]
        pred_month = usable_months[i + 1]
        train_end = train_month.end_time.date()
        pred_mask = df_dates.dt.to_period('M') == pred_month
        pred_indices = df.index[pred_mask][:pred_days]
        if len(pred_indices) < 3:
            continue
        periods.append((train_end, pred_indices.tolist()))
    return periods

def _train_single_model(model_type, window_label, train_df, market, active_groups):
    window_draws = get_window_size(market, window_label)
    sliced_df = slice_window(train_df, window_draws)
    if len(sliced_df) < 10:
        return None, None
    feature_df, y, group_cols = build_features(sliced_df, active_groups)
    feature_cols = [c for c in feature_df.columns if c != '_date']
    X = feature_df[feature_cols].values
    if len(X) < 5:
        return None, None
    model_cls = MODEL_TYPES[model_type]
    model = model_cls()
    model.fit(X, y.values)
    return model, feature_cols

def _compute_metrics(predictions, actuals):
    n = len(predictions)
    if n == 0: return None
    top1, top4, top10 = 0, 0, 0
    brier_sum = 0.0
    for probs, act in zip(predictions, actuals):
        act = int(act)
        # Brier
        onehot = np.zeros(100)
        onehot[act] = 1.0
        brier_sum += np.mean((probs - onehot)**2)
        
        sorted_indices = np.argsort(probs)[::-1]
        if act == sorted_indices[0]: top1 += 1
        if act in sorted_indices[:4]: top4 += 1
        if act in sorted_indices[:10]: top10 += 1
        
    return {
        'top1': top1 / n,
        'top4': top4 / n,
        'top10': top10 / n,
        'brier': brier_sum / n,
        'raw_top1': top1,
        'raw_n': n
    }

def run_walk_forward(market: str, active_groups: list = None, num_periods: int = 4, pred_days: int = 7, verbose: bool = True):
    if active_groups is None:
        active_groups = ALL_FEATURE_GROUPS.copy()

    df = load_raw_data(market)
    periods = _get_validation_periods(df, num_periods, pred_days)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  V4 JODI WALK-FORWARD VALIDATION: {market.upper()}")
        print(f"  {len(periods)} periods × {pred_days} prediction days")
        print(f"{'='*70}\n")

    all_metrics = defaultdict(list)
    window_labels = ['3m', '6m', '12m', 'full']

    for period_idx, (train_end, pred_indices) in enumerate(periods):
        if verbose:
            print(f"  Period {period_idx+1}: Train up to {train_end}, predict {len(pred_indices)} days")

        train_mask = pd.to_datetime(df['Date']).dt.date <= train_end
        train_df = df[train_mask].copy()

        for wl in window_labels:
            for mt in MODEL_TYPES.keys():
                model_id = f"{wl}_{mt}"
                model, feat_cols = _train_single_model(mt, wl, train_df, market, active_groups)
                if model is None:
                    continue

                predictions = []
                actuals = []

                for pred_idx in pred_indices:
                    actual_jodi = int(df.iloc[pred_idx]['Morning_number']) * 10 + int(df.iloc[pred_idx]['Evening_number'])
                    context_df = df.iloc[:pred_idx].copy()
                    window_draws = get_window_size(market, wl)
                    context_sliced = slice_window(context_df, window_draws)
                    if len(context_sliced) < 5: continue
                    feat_df, _, _ = build_features(context_sliced, active_groups)
                    if len(feat_df) == 0: continue
                    last_row_feats = feat_df.iloc[[-1]]
                    feat_only = [c for c in last_row_feats.columns if c != '_date']
                    X_pred = last_row_feats[feat_only].values

                    probs = model.predict_proba(X_pred)
                    predictions.append(probs)
                    actuals.append(actual_jodi)

                if len(predictions) >= 3:
                    metrics = _compute_metrics(predictions, actuals)
                    if metrics:
                        all_metrics[model_id].append(metrics)

    avg_metrics = {}
    for model_id, metric_list in all_metrics.items():
        avg = {}
        for key in metric_list[0].keys():
            avg[key] = np.mean([m[key] for m in metric_list])
        avg['n_periods'] = len(metric_list)
        # Sum raw counts for financial backtest
        avg['total_hits'] = sum(m['raw_top1'] for m in metric_list)
        avg['total_days'] = sum(m['raw_n'] for m in metric_list)
        avg_metrics[model_id] = avg

    if verbose:
        _print_metrics_table(avg_metrics)

    return avg_metrics

def _print_metrics_table(avg_metrics):
    print(f"\n{'='*95}")
    print(f"  {'Model':<12} {'Top-1':>7} {'Top-4':>7} {'Top-10':>7} {'Brier':>8} {'Hits/Days':>10} {'ROI (50rs)':>12}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*10} {'-'*12}")

    for model_id in sorted(avg_metrics.keys()):
        m = avg_metrics[model_id]
        total_cost = m['total_days'] * 50
        total_payout = m['total_hits'] * 4750
        roi = total_payout - total_cost
        roi_str = f"+₹{roi}" if roi >= 0 else f"-₹{abs(roi)}"
        print(f"  {model_id:<12} {m['top1']:>6.1%} {m['top4']:>6.1%} {m['top10']:>6.1%} "
              f"{m['brier']:>8.5f} {m['total_hits']:>3}/{m['total_days']:<5} {roi_str:>12}")
    print(f"{'='*95}")
    print(f"  Random baseline: Top-1 = 1.0%, Top-4 = 4.0%, Top-10 = 10.0%")

def learn_weights(avg_metrics, temperature=0.01):
    model_ids = sorted(avg_metrics.keys())
    if not model_ids: return {}
    top1_scores = np.array([avg_metrics[mid]['top1'] for mid in model_ids])
    raw_scores = top1_scores / temperature
    raw_scores -= raw_scores.max()
    exp_scores = np.exp(raw_scores)
    weights = exp_scores / exp_scores.sum()
    return {mid: float(w) for mid, w in zip(model_ids, weights)}

def prune_models(weights, cumulative_threshold=0.95):
    if not weights: return {}
    sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    survivors = {}
    cumulative = 0.0
    for mid, w in sorted_models:
        survivors[mid] = w
        cumulative += w
        if cumulative >= cumulative_threshold:
            break
    total = sum(survivors.values())
    return {mid: w / total for mid, w in survivors.items()}

def run_feature_ablation(market, base_groups=None, verbose=True):
    if base_groups is None:
        base_groups = ALL_FEATURE_GROUPS.copy()
    if verbose:
        print(f"\n{'='*70}")
        print(f"  V4 FEATURE ABLATION: {market.upper()}")
        print(f"{'='*70}\n")
        print("  Running baseline with ALL feature groups...")
    baseline_metrics = run_walk_forward(market, base_groups, verbose=False)
    baseline_brier = np.mean([m['brier'] for m in baseline_metrics.values()])
    if verbose:
        print(f"  Baseline avg Brier: {baseline_brier:.5f}\n")
    groups_to_remove = []
    for group in base_groups:
        reduced_groups = [g for g in base_groups if g != group]
        if verbose:
            print(f"  Testing WITHOUT '{group}'...", end=' ')
        reduced_metrics = run_walk_forward(market, reduced_groups, verbose=False)
        if not reduced_metrics:
            if verbose: print("SKIP")
            continue
        reduced_brier = np.mean([m['brier'] for m in reduced_metrics.values()])
        diff = reduced_brier - baseline_brier
        if diff <= 0:
            if verbose: print(f"Brier={reduced_brier:.5f} (Δ={diff:+.5f}) → REMOVE ❌")
            groups_to_remove.append(group)
        else:
            if verbose: print(f"Brier={reduced_brier:.5f} (Δ={diff:+.5f}) → KEEP ✅")
    surviving_groups = [g for g in base_groups if g not in groups_to_remove]
    if verbose:
        print(f"\n  Surviving features: {surviving_groups}")
    return surviving_groups

def save_state(market, weights, surviving_groups, avg_metrics):
    _ensure_state_dir()
    state = {
        'market': market,
        'weights': weights,
        'surviving_groups': surviving_groups,
        'timestamp': datetime.now().isoformat(),
        'model_metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in avg_metrics.items()},
    }
    path = os.path.join(STATE_DIR, f'{market}_state.json')
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"\n  State saved to {path}")

def load_state(market):
    path = os.path.join(STATE_DIR, f'{market}_state.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)
