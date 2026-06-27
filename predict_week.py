import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
import warnings
from pipeline import load_and_clean_data, preprocess_and_engineer, get_lunar_phase

warnings.filterwarnings('ignore')

def train_full_models(df):
    exclude_cols = ['Date', 'Day_of_Week', 'Morning_number', 'Evening_number']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features]
    y_m = df['Morning_number'].astype(int)
    y_e = df['Evening_number'].astype(int)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 10,
        'eval_metric': 'mlogloss',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1  # Use all cores to train instantly!
    }
    
    print("Training models instantly on all historical data (Bypassing walk-forward)...")
    model_m = xgb.XGBClassifier(**params)
    model_e = xgb.XGBClassifier(**params)
    
    model_m.fit(X, y_m)
    model_e.fit(X, y_e)
    
    return model_m, model_e, X, features

def predict_future_days(model_m, model_e, processed_df, X, features, target_dates):
    print("\n=======================================================")
    print(f"--- GENERATING SEQUENTIAL PREDICTIONS FOR UP TO TODAY ---")
    print("=======================================================")
    
    in_holidays = holidays.IN(years=[2026])
    diwali_dates = [date for date, name in in_holidays.items() if 'Diwali' in name]
    
    def is_festival_window(date_obj):
        if date_obj in in_holidays:
            return 1
        for d_date in diwali_dates:
            if abs((date_obj.date() - d_date).days) <= 3:
                return 1
        return 0

    # Get the state of the very last known draw
    current_state = X.iloc[[-1]].copy()
    last_date = processed_df['Date'].iloc[-1]
    
    last_m_outcome = processed_df['Morning_number'].iloc[-1]
    last_e_outcome = processed_df['Evening_number'].iloc[-1]
    
    # We will generate predictions day by day and feed them back as lags
    for i in range(1, target_dates + 1):
        target_date = last_date + pd.Timedelta(days=i)
        
        # Advance state
        current_state['Draw_Index'] += 1
        
        # Shift Lags correctly
        for lag in [3, 2]:
            current_state[f'Morning_lag_{lag}'] = current_state[f'Morning_lag_{lag-1}']
            current_state[f'Evening_lag_{lag}'] = current_state[f'Evening_lag_{lag-1}']
            
        current_state['Morning_lag_1'] = last_m_outcome
        current_state['Evening_lag_1'] = last_e_outcome
        
        # Recompute exact cultural bias for this new date
        current_state['Is_Festival'] = is_festival_window(target_date)
        current_state['Is_Amavasya_Purnima'] = get_lunar_phase(target_date)
        current_state['Days_Since_Last_Draw'] = 1 
        
        # Predict
        m_probs = model_m.predict_proba(current_state)[0]
        e_probs = model_e.predict_proba(current_state)[0]
        
        top_m = np.argmax(m_probs)
        top_e = np.argmax(e_probs)
        
        date_str = target_date.strftime('%Y-%m-%d')
        day_str = target_date.strftime('%A')
        print(f"\n>> PREDICTION FOR: {date_str} ({day_str})")
        
        # Sort and show top 3 probabilities to give context
        m_top3 = np.argsort(m_probs)[-3:][::-1]
        e_top3 = np.argsort(e_probs)[-3:][::-1]
        
        print(f"  Morning Top 3: " + " | ".join([f"Num {num} ({m_probs[num]*100:.1f}%)" for num in m_top3]))
        print(f"  Evening Top 3: " + " | ".join([f"Num {num} ({e_probs[num]*100:.1f}%)" for num in e_top3]))
        
        # Use the highest probability number as the actual result to build tomorrow's lag
        last_m_outcome = top_m
        last_e_outcome = top_e

if __name__ == "__main__":
    filepath = 'combined_data.csv'
    df = load_and_clean_data(filepath)
    processed_df = preprocess_and_engineer(df)
    
    # Train extremely quickly
    model_m, model_e, X, features = train_full_models(processed_df)
    
    # Generate 5 days of predictions (June 22 to June 26)
    predict_future_days(model_m, model_e, processed_df, X, features, target_dates=5)
