import pandas as pd
import numpy as np
from hybrid_1.predict import predict_tomorrow

# The actual results that occurred after July 2nd:
# We will feed these to the model day by day and see what it predicted.
actuals = [
    {'date': '2026-07-03', 'dow': 'Fri', 'open_panna': '1,4,5', 'open': 0, 'close_panna': '1,2,0', 'close': 3},
    {'date': '2026-07-04', 'dow': 'Sat', 'open_panna': '3,4,7', 'open': 4, 'close_panna': '1,1,7', 'close': 9},
    {'date': '2026-07-06', 'dow': 'Mon', 'open_panna': '4,7,8', 'open': 9, 'close_panna': '2,4,9', 'close': 5},
    {'date': '2026-07-07', 'dow': 'Tue', 'open_panna': '5,6,6', 'open': 7, 'close_panna': '3,5,7', 'close': 5},
    {'date': '2026-07-08', 'dow': 'Wed', 'open_panna': '2,7,7', 'open': 6, 'close_panna': '4,7,8', 'close': 9},
    {'date': '2026-07-09', 'dow': 'Thu', 'open_panna': '5,7,9', 'open': 1, 'close_panna': '3,4,7', 'close': 4},
]

def simulate_days():
    # Read the dataset (which currently ends on 2026-07-02)
    file_path = 'true_kalyan_main_dataset.csv'
    
    # We will backup the original
    import shutil
    shutil.copy(file_path, file_path + '.backup')

    try:
        for i in range(len(actuals)):
            act = actuals[i]
            
            # Predict for this day BEFORE adding it to the dataset
            print(f"\n{'='*60}")
            print(f" PREDICTING FOR: {act['date']} ({act['dow']})")
            print(f"{'='*60}")
            
            # Run the prediction
            pred_result = predict_tomorrow('kalyan', verbose=False)
            if pred_result is None:
                print("Failed to run prediction.")
                continue
            
            m_probs = pred_result['morning_probs']
            e_probs = pred_result['evening_probs']
            
            top_m = np.argsort(m_probs)[::-1][:3]
            top_e = np.argsort(e_probs)[::-1][:3]
            
            actual_m = act['open']
            actual_e = act['close']
            
            m_hit = actual_m in top_m
            e_hit = actual_e in top_e
            
            print(f"Predicted Morning (Top 3): {top_m.tolist()} | Actual: {actual_m} -> {'HIT ✅' if m_hit else 'MISS ❌'}")
            print(f"Predicted Evening (Top 3): {top_e.tolist()} | Actual: {actual_e} -> {'HIT ✅' if e_hit else 'MISS ❌'}")
            print(f"Actual Jodi: {actual_m}{actual_e}")
            
            # Now ADD this day to the dataset so the model can use it for the next day!
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Extract last index
            last_line = lines[-1].strip()
            if not last_line:
                last_line = lines[-2].strip()
            last_idx = int(last_line.split(',')[-1])
            new_idx = last_idx + 1
            
            op = act['open_panna'].split(',')
            cp = act['close_panna'].split(',')
            
            new_line = f"{act['date']},{act['dow']},{op[0]},{op[1]},{op[2]},{act['open']},{cp[0]},{cp[1]},{cp[2]},{act['close']},{new_idx}\n"
            
            with open(file_path, 'a') as f:
                f.write(new_line)
                
        # Now predict for TOMORROW (Friday 10/07/2026)
        print(f"\n{'='*60}")
        print(f" PREDICTING FOR TODAY/TOMORROW (After all updates)")
        print(f"{'='*60}")
        predict_tomorrow('kalyan', verbose=True)

    finally:
        # Restore the original dataset so we don't mess up their files!
        import os
        if os.path.exists(file_path + '.backup'):
            shutil.copy(file_path + '.backup', file_path)
            os.remove(file_path + '.backup')

if __name__ == '__main__':
    simulate_days()
