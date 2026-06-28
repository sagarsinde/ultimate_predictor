import pandas as pd
import numpy as np
import joblib
import json
from feature_engineering import generate_features
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    print("--- Phase 3: Production Inference Engine ---")
    
    try:
        print("Loading Calibrated AI Models...")
        model_m = joblib.load('morning_model.joblib')
        model_e = joblib.load('evening_model.joblib')
        with open('model_features.json', 'r') as f:
            features = json.load(f)
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_model.py first!")
        return

    print("Loading raw dataset to reconstruct exact mathematical state...")
    df = pd.read_csv('true_kalyan_main_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Generate features for the entire history to guarantee PageRank/FFT stability
    # This takes < 1 second in Pandas
    print("Reconstructing Causal Feature Universe...")
    featured_df = generate_features(df)
    
    # We want to predict "Tomorrow"
    # To do this, we take the very last known row and shift its state forward 1 day
    last_row = featured_df.iloc[[-1]].copy()
    
    last_actual_date = last_row['Date'].dt.date.values[0]
    # KALYAN market runs Mon-Sat. If today is Saturday, next is Monday (+2 days)
    # Note: Pandas dt.dayofweek -> Mon=0, Sun=6
    current_dow = pd.to_datetime(last_actual_date).dayofweek
    days_to_add = 2 if current_dow == 5 else 1
    
    pred_date = pd.to_datetime(last_actual_date) + timedelta(days=days_to_add)
    pred_date_str = pred_date.strftime('%Y-%m-%d (%A)')
    
    print(f"\nSynthesizing State for Target Date: {pred_date_str}")
    
    # Create the prediction vector using the exact same columns the model was trained on
    X_predict = last_row[features].copy()
    
    # The Lags need to shift. What was "Today's Result" becomes "Lag 1" for tomorrow
    # Since we are predicting tomorrow, we must manually roll the Lags forward 1 step
    for i in range(7, 1, -1):
        if f'Morning_lag_{i}' in X_predict.columns:
            X_predict[f'Morning_lag_{i}'] = X_predict[f'Morning_lag_{i-1}']
            X_predict[f'Evening_lag_{i}'] = X_predict[f'Evening_lag_{i-1}']
            
    # Lag 1 becomes the actual result from the last known row
    X_predict['Morning_lag_1'] = df.iloc[-1]['Morning_number']
    X_predict['Evening_lag_1'] = df.iloc[-1]['Evening_number']
    
    # Update Time Cyclicals for Tomorrow
    next_dow_num = pred_date.dayofweek if pred_date.dayofweek < 6 else 0 # Fallback mapping
    X_predict['DOW_sin'] = np.sin(2 * np.pi * next_dow_num / 6.0)
    X_predict['DOW_cos'] = np.cos(2 * np.pi * next_dow_num / 6.0)
    
    # Update FFT Waves for next step
    next_step = len(df)
    if 'FFT_Wave_2_2' in X_predict.columns:
        X_predict['FFT_Wave_2_2'] = np.sin(2 * np.pi * next_step / 2.2)
        X_predict['FFT_Wave_4_9'] = np.sin(2 * np.pi * next_step / 4.9)
    
    print("\nExecuting Calibrated Probability Matrix...")
    m_probs = model_m.predict_proba(X_predict)[0]
    e_probs = model_e.predict_proba(X_predict)[0]
    
    # Joint Probability Matrix
    jodi_probs = {}
    for mm in range(10):
        for ee in range(10):
            jodi_probs[f"{mm}{ee}"] = m_probs[mm] * e_probs[ee]
            
    top4_jodis = sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]
    m_top3 = np.argsort(m_probs)[-3:][::-1]
    e_top3 = np.argsort(e_probs)[-3:][::-1]
    
    print("\n=======================================================")
    print(f"   ISOTONIC-CALIBRATED PREDICTIONS: {pred_date_str}")
    print("=======================================================")
    print("Morning Draw - Top 3 Probabilities:")
    for num in m_top3:
        print(f"  Digit {num}: {m_probs[num]*100:.2f}%")
        
    print("\nEvening Draw - Top 3 Probabilities:")
    for num in e_top3:
        print(f"  Digit {num}: {e_probs[num]*100:.2f}%")
        
    print("\nTop 4 Jodi Predictions:")
    jodi_str = ", ".join([f"{j[0]} ({j[1]*100:.1f}%)" for j in top4_jodis])
    print(f"  Top 4 Jodi: {jodi_str}")
    print("=======================================================\n")

if __name__ == '__main__':
    main()
