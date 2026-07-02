"""
v2/run_predict.py — Daily Prediction

Usage: python -m v2.run_predict kalyan
       python -m v2.run_predict mb

Loads learned state from the last backtest, trains surviving models
on current data, and outputs the weighted ensemble prediction.
"""

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.predict import predict_tomorrow
from v2.validator import load_state


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m v2.run_predict [kalyan|mb]")
        sys.exit(1)

    market = sys.argv[1].lower()
    if market not in ('kalyan', 'mb'):
        print(f"Unknown market: {market}. Use 'kalyan' or 'mb'.")
        sys.exit(1)

    # Check if state exists
    state = load_state(market)
    if state is None:
        print(f"No saved state for {market}.")
        print(f"Run 'python -m v2.run_backtest {market}' first to build the ensemble.")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"  v2 SELF-IMPROVING ENSEMBLE — PREDICTION: {market.upper()}")
    print(f"{'#'*70}")

    predict_tomorrow(market, verbose=True)


if __name__ == '__main__':
    main()
