"""
Instantly recalculate weights without re-running backtest.
Run this after adding new daily data to recalculate Brier Scores.
"""

import sys
import os
import json

# Add the market directory to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.validator import learn_weights, prune_models

MARKET_NAME = "MADHUR"

def update_market_weights():
    state_path = os.path.join(os.path.dirname(__file__), 'models', 'state.json')
    
    if not os.path.exists(state_path):
        print(f"Error: {state_path} does not exist. Run backtest first.")
        return

    with open(state_path, 'r') as f:
        state = json.load(f)

    if 'model_metrics' not in state:
        print("Error: No model_metrics in state file.")
        return

    print(f"\nRecalculating weights for {MARKET_NAME} using Top-3 Accuracy...")
    
    # Recalculate weights using the new Top-3 logic
    raw_weights = learn_weights(state['model_metrics'])
    
    print("\nNew Raw Weights:")
    for mid, w in sorted(raw_weights.items(), key=lambda x: -x[1]):
        print(f"  {mid:<15} {w:.4f}")

    pruned_weights = prune_models(raw_weights, cumulative_threshold=0.95)
    
    print(f"\nSurviving Models ({len(pruned_weights)}):")
    for mid, w in sorted(pruned_weights.items(), key=lambda x: -x[1]):
        print(f"  {mid:<15} {w:.4f} (renormalized)")

    # Save back to state
    state['weights'] = pruned_weights
    
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
        
    print(f"\n✅ Weights updated and saved successfully! No backtest needed.")

if __name__ == '__main__':
    update_market_weights()
