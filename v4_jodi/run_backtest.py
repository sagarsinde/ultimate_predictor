import sys
from v4_jodi.validator import run_feature_ablation, run_walk_forward, learn_weights, prune_models, save_state

def main():
    market = 'kalyan'
    if len(sys.argv) > 1:
        market = sys.argv[1].lower()

    print(f"\n{'#'*70}")
    print(f"  V4 JODI ENSEMBLE — FULL BACKTEST: {market.upper()}")
    print(f"{'#'*70}\n")

    print("[STEP 1/4] Feature Ablation...")
    surviving_groups = run_feature_ablation(market, verbose=True)

    print("\n[STEP 2/4] Walk-Forward Validation with surviving features...")
    avg_metrics = run_walk_forward(market, active_groups=surviving_groups, verbose=True)

    print("\n[STEP 3/4] Learning model weights from Top-1 Accuracy...")
    weights = learn_weights(avg_metrics)
    print("\n  Raw weights (before pruning):")
    for mid, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    {mid:<15} {w:.4f}")

    print("\n[STEP 4/4] Pruning weak models (95% cumulative weight)...")
    surviving_weights = prune_models(weights, 0.95)
    print(f"\n  Surviving: {len(surviving_weights)} models")
    print(f"  Pruned: {len(weights) - len(surviving_weights)} models")
    for mid, w in sorted(surviving_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    {mid:<15} {w:.4f} (renormalized)")

    save_state(market, surviving_weights, surviving_groups, avg_metrics)

    print(f"\n{'#'*70}")
    print(f"  BACKTEST COMPLETE for {market.upper()}")
    print(f"  Now run: python -m v4_jodi.run_predict {market}")
    print(f"{'#'*70}\n")

if __name__ == '__main__':
    main()
