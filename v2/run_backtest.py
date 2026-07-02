"""
v2/run_backtest.py — Full Validation + Feature Ablation + Model Pruning

Usage: python -m v2.run_backtest kalyan
       python -m v2.run_backtest mb

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

from v2.validator import (
    run_walk_forward, learn_weights, prune_models,
    build_calibration, run_feature_ablation, save_state,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m v2.run_backtest [kalyan|mb]")
        sys.exit(1)

    market = sys.argv[1].lower()
    if market not in ('kalyan', 'mb'):
        print(f"Unknown market: {market}. Use 'kalyan' or 'mb'.")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"  v2 SELF-IMPROVING ENSEMBLE — FULL BACKTEST: {market.upper()}")
    print(f"{'#'*70}")

    # Step 1: Feature ablation
    print("\n[STEP 1/5] Feature Ablation...")
    surviving_groups = run_feature_ablation(market, verbose=True)

    # Step 2: Walk-forward validation with surviving features
    print("\n[STEP 2/5] Walk-Forward Validation with surviving features...")
    avg_metrics, (cal_data_m, cal_data_e) = run_walk_forward(
        market, active_groups=surviving_groups, verbose=True
    )

    if not avg_metrics:
        print("ERROR: No valid model metrics produced. Check your data.")
        sys.exit(1)

    # Step 3: Learn weights
    print("\n[STEP 3/5] Learning model weights from Brier Scores...")
    raw_weights = learn_weights(avg_metrics)

    print(f"\n  Raw weights (before pruning):")
    for mid, w in sorted(raw_weights.items(), key=lambda x: -x[1]):
        print(f"    {mid:<15} {w:.4f}")

    # Step 4: Prune weak models
    print("\n[STEP 4/5] Pruning weak models (95% cumulative weight)...")
    pruned_weights = prune_models(raw_weights, cumulative_threshold=0.95)

    pruned_count = len(raw_weights) - len(pruned_weights)
    print(f"\n  Surviving: {len(pruned_weights)} models")
    print(f"  Pruned: {pruned_count} models")
    for mid, w in sorted(pruned_weights.items(), key=lambda x: -x[1]):
        print(f"    {mid:<15} {w:.4f} (renormalized)")

    # Step 5: Build confidence calibration
    print("\n[STEP 5/5] Building confidence calibration...")
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
        market, pruned_weights, surviving_groups,
        calibrator_m, calibrator_e,
        thresholds_m, thresholds_e,
        avg_metrics,
    )

    print(f"\n{'#'*70}")
    print(f"  BACKTEST COMPLETE for {market.upper()}")
    print(f"  Now run: python -m v2.run_predict {market}")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
