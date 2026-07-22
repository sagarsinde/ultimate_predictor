"""
v2/run_backtest.py — Full Validation + Feature Ablation + Model Pruning

Usage: python -m hybrid_1_1.run_backtest kalyan
       python -m hybrid_1_1.run_backtest mb
       python -m hybrid_1_1.run_backtest madhur

This is the first-time setup command. Run this once (or periodically) to:
  1. Run feature ablation to find which features actually help
  2. Run walk-forward validation with surviving features
  3. Learn model weights from Brier Scores
  4. Prune weak models
  5. Build confidence calibration
  6. Save all learned state to disk
"""

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_1_1.validator import (
    run_walk_forward, learn_weights, prune_models,
    build_calibration, run_feature_ablation, save_state,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hybrid_1_1.run_backtest [kalyan|mb|madhur]")
        sys.exit(1)

    market = sys.argv[1].lower()
    if market not in ('kalyan', 'mb', 'madhur'):
        print(f"Unknown market: {market}. Use 'kalyan', 'mb', or 'madhur'.")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"  v2 SELF-IMPROVING ENSEMBLE — FULL BACKTEST: {market.upper()}")
    print(f"{'#'*70}")

    # Step 1: Feature ablation
    print("\n[STEP 1/5] Feature Ablation...")
    surviving_groups = run_feature_ablation(market, verbose=True)

    # Step 2: Walk-forward validation with surviving features
    print("\n[STEP 2/6] Walk-Forward Validation with surviving features...")
    avg_metrics, (cal_data_m, cal_data_e) = run_walk_forward(
        market, active_groups=surviving_groups, verbose=True
    )

    if not avg_metrics:
        print("ERROR: No valid model metrics produced. Check your data.")
        sys.exit(1)

    # Step 3: Learn weights
    print("\n[STEP 3/6] Learning model weights from Top-3 Accuracy...")
    raw_weights_dict = learn_weights(avg_metrics)

    print(f"\n  Raw weights (Morning):")
    for mid, w in sorted(raw_weights_dict['weights_m'].items(), key=lambda x: -x[1]):
        print(f"    {mid:<15} {w:.4f}")

    print(f"\n  Raw weights (Evening):")
    for mid, w in sorted(raw_weights_dict['weights_e'].items(), key=lambda x: -x[1]):
        print(f"    {mid:<15} {w:.4f}")

    # Step 4: Prune weak models
    print("\n[STEP 4/6] Pruning weak models (95% cumulative weight)...")
    pruned_weights_dict = prune_models(raw_weights_dict, cumulative_threshold=0.95)

    print(f"\n  Surviving Morning Models: {len(pruned_weights_dict['weights_m'])}")
    for mid, w in sorted(pruned_weights_dict['weights_m'].items(), key=lambda x: -x[1]):
        print(f"    {mid:<15} {w:.4f} (renormalized)")

    print(f"\n  Surviving Evening Models: {len(pruned_weights_dict['weights_e'])}")
    for mid, w in sorted(pruned_weights_dict['weights_e'].items(), key=lambda x: -x[1]):
        print(f"    {mid:<15} {w:.4f} (renormalized)")

    # Step 5: Build confidence calibration
    print("\n[STEP 5/6] Building confidence calibration...")
    calibrator_m, thresholds_m = build_calibration(cal_data_m)
    calibrator_e, thresholds_e = build_calibration(cal_data_e)

    print(f"\n  Morning thresholds: {thresholds_m}")
    print(f"  Evening thresholds: {thresholds_e}")

    if calibrator_m is None:
        print("  WARNING: Not enough calibration data for morning. Using fallback thresholds.")
    if calibrator_e is None:
        print("  WARNING: Not enough calibration data for evening. Using fallback thresholds.")

    # Save everything
    save_state(
        market, pruned_weights_dict, surviving_groups,
        calibrator_m, calibrator_e,
        thresholds_m, thresholds_e,
        avg_metrics,
    )

    print(f"\n[STEP 6/6] Training & Saving Winning Models to Disk...")
    from hybrid_1_1.validator import _train_single_model
    from hybrid_1_1.features import load_raw_data
    df = load_raw_data(market)
    
    active_m = set(pruned_weights_dict['weights_m'].keys())
    active_e = set(pruned_weights_dict['weights_e'].keys())
    active_models = active_m | active_e
    
    dir_path = os.path.join("trained_models", "backtest_winners", market)
    os.makedirs(dir_path, exist_ok=True)
    
    for model_id in sorted(active_models):
        parts = model_id.split('_', 1)
        window_label, model_type = parts[0], parts[1]
        
        print(f"  -> Retraining {model_id} on latest data...")
        model_m_obj, model_e_obj, _, _ = _train_single_model(
            model_type, window_label, df, market, surviving_groups
        )
        
        if model_m_obj and model_id in active_m:
            model_m_obj.save_models(dir_path, f"{model_id}_morning")
        if model_e_obj and model_id in active_e:
            model_e_obj.save_models(dir_path, f"{model_id}_evening")

    print(f"\n{'#'*70}")
    print(f"  BACKTEST COMPLETE for {market.upper()}")
    print(f"  Now run: python -m hybrid_1_1.run_predict {market}")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
