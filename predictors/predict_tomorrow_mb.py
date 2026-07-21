import pandas as pd
import numpy as np
import joblib
import json
from feature_engineering_mb import generate_features
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("   MAIN BAZAR - TOMORROW'S PREDICTION ENGINE")
    print("=" * 60)
    
    # 1. Load Models
    try:
        print("\nLoading Calibrated AI Models...")
        model_m = joblib.load('mb_morning_model.joblib')
        model_e = joblib.load('mb_evening_model.joblib')
        with open('mb_model_features.json', 'r') as f:
            features = json.load(f)
    except FileNotFoundError:
        print("Error: Model files not found. Run train_model_mb.py first!")
        return

    # 2. Load raw dataset and generate features
    print("Loading raw dataset...")
    df = pd.read_csv('main_bazar_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("Reconstructing Causal Feature Universe...")
    featured_df = generate_features(df)
    
    # 3. Determine prediction date
    last_row = featured_df.iloc[[-1]].copy()
    last_actual_date = last_row['Date'].dt.date.values[0]
    
    # Main Bazar runs Mon-Fri. Skip weekends.
    current_dow = pd.to_datetime(last_actual_date).dayofweek  # Mon=0, Sun=6
    if current_dow == 4:    # Friday -> Monday (+3)
        days_to_add = 3
    elif current_dow == 5:  # Saturday -> Monday (+2)
        days_to_add = 2
    elif current_dow == 6:  # Sunday -> Monday (+1)
        days_to_add = 1
    else:
        days_to_add = 1     # Normal weekday -> next day
    
    pred_date = pd.to_datetime(last_actual_date) + timedelta(days=days_to_add)
    pred_date_str = pred_date.strftime('%Y-%m-%d (%A)')
    
    print(f"\nTarget Prediction Date: {pred_date_str}")
    
    # 4. Build prediction vector
    X_predict = last_row[features].copy()
    
    # Roll lags forward
    for i in range(7, 1, -1):
        if f'Morning_lag_{i}' in X_predict.columns:
            X_predict[f'Morning_lag_{i}'] = X_predict[f'Morning_lag_{i-1}']
            X_predict[f'Evening_lag_{i}'] = X_predict[f'Evening_lag_{i-1}']
            
    X_predict['Morning_lag_1'] = df.iloc[-1]['Morning_number']
    X_predict['Evening_lag_1'] = df.iloc[-1]['Evening_number']
    
    # Update time cyclicals for tomorrow
    next_dow_num = pred_date.dayofweek
    X_predict['DOW_sin'] = np.sin(2 * np.pi * next_dow_num / 5.0)
    X_predict['DOW_cos'] = np.cos(2 * np.pi * next_dow_num / 5.0)
    
    # Update FFT waves
    next_step = len(df)
    if 'FFT_Wave_2_2' in X_predict.columns:
        X_predict['FFT_Wave_2_2'] = np.sin(2 * np.pi * next_step / 2.2)
        X_predict['FFT_Wave_4_9'] = np.sin(2 * np.pi * next_step / 4.9)
    
    # 5. Generate probabilities
    print("Executing Calibrated Probability Matrix...")
    X_predict = X_predict.astype(float)
    m_probs = model_m.predict_proba(X_predict)[0]
    e_probs = model_e.predict_proba(X_predict)[0]
    
    # 6. Confidence check (17% threshold for Main Bazar)
    max_m_conf = np.max(m_probs)
    max_e_conf = np.max(e_probs)
    
    # Joint Jodi probability
    jodi_probs = {}
    for mm in range(10):
        for ee in range(10):
            jodi_probs[f"{mm}{ee}"] = m_probs[mm] * e_probs[ee]
            
    top4_jodis = sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]
    m_top3 = np.argsort(m_probs)[-3:][::-1]
    e_top3 = np.argsort(e_probs)[-3:][::-1]
    
    # 7. Display results
    print("\n" + "=" * 60)
    print(f"   MAIN BAZAR PREDICTIONS: {pred_date_str}")
    print("=" * 60)
    
    print("\n--- OPEN (Morning) Draw - Top 3 ---")
    m_signal = "HIGH CONFIDENCE" if max_m_conf >= 0.17 else "LOW CONFIDENCE"
    print(f"    AI Confidence: {max_m_conf*100:.1f}% [{m_signal}]")
    for num in m_top3:
        bar = "█" * int(m_probs[num] * 100)
        print(f"    Digit {num}: {m_probs[num]*100:.2f}%  {bar}")
        
    print("\n--- CLOSE (Evening) Draw - Top 3 ---")
    e_signal = "HIGH CONFIDENCE" if max_e_conf >= 0.17 else "LOW CONFIDENCE"
    print(f"    AI Confidence: {max_e_conf*100:.1f}% [{e_signal}]")
    for num in e_top3:
        bar = "█" * int(e_probs[num] * 100)
        print(f"    Digit {num}: {e_probs[num]*100:.2f}%  {bar}")
        
    print("\n--- Top 4 JODI Predictions ---")
    for rank, (jodi, prob) in enumerate(top4_jodis, 1):
        bar = "█" * int(prob * 200)
        print(f"    #{rank}  Jodi {jodi}: {prob*100:.2f}%  {bar}")
    
    # 8. Betting recommendation
    print("\n" + "-" * 60)
    print("   BETTING RECOMMENDATION (17% Confidence Filter)")
    print("-" * 60)
    
    play_morning = max_m_conf >= 0.17
    play_evening = max_e_conf >= 0.17
    play_jodi = play_morning and play_evening
    
    if play_morning:
        print(f"   ✅ OPEN:  BET on digits {', '.join(str(d) for d in m_top3)}")
    else:
        print(f"   ❌ OPEN:  SKIP (confidence {max_m_conf*100:.1f}% < 17%)")
        
    if play_evening:
        print(f"   ✅ CLOSE: BET on digits {', '.join(str(d) for d in e_top3)}")
    else:
        print(f"   ❌ CLOSE: SKIP (confidence {max_e_conf*100:.1f}% < 17%)")
        
    if play_jodi:
        top_jodi = top4_jodis[0][0]
        print(f"   ✅ JODI:  BET on {', '.join(j[0] for j in top4_jodis)}")
    else:
        print(f"   ❌ JODI:  SKIP (need both Open & Close confidence)")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
