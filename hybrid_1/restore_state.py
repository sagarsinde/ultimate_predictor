import json
import os

STATE_DIR = os.path.join(os.path.dirname(__file__), 'state')
os.makedirs(STATE_DIR, exist_ok=True)

state = {
    "market": "kalyan",
    "weights_m": {
        "1m_freq": 0.075,
        "1m_rf": 0.075,
        "2m_cat": 0.313,
        "2m_xgb": 0.313,
        "3m_cat": 0.047,
        "3m_rf": 0.075,
        "full_dowfreq": 0.075,
        "full_freq": 0.029
    },
    "weights_e": {
        "1m_xgb": 0.041,
        "2m_cat": 0.041,
        "2m_rf": 0.172,
        "2m_xgb": 0.719,
        "full_cat": 0.026
    },
    "surviving_groups": [
        "hits_last_7",
        "day_of_week",
        "morning_evening_corr",
        "hot_cold_ratio",
        "gap_velocity"
    ],
    "thresholds_m": {"strong": 1.0, "good": 1.0, "marginal": 1.0},
    "thresholds_e": {"strong": 1.0, "good": 1.0, "marginal": 0.2955},
    "timestamp": "2026-07-09T16:35:19"
}

path = os.path.join(STATE_DIR, 'kalyan_state.json')
with open(path, 'w') as f:
    json.dump(state, f, indent=4)

print(f"Successfully restored state file to {path}!")
print("You can now run: !python -m hybrid_1.cascade_predict kalyan")
