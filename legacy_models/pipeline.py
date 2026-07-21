import os
import pandas as pd
import numpy as np
import holidays
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

try:
    from panchang import Location
    # Some panchang versions have panchang.compute, some have different APIs
    # We will try to import compute or fallback to a custom lunar phase calculation
    try:
        from panchang import panchang as panchang_calc
    except ImportError:
        panchang_calc = None
except ImportError:
    Location = None
    panchang_calc = None

def get_lunar_phase(date_obj):
    """
    Returns True if Amavasya or Purnima, False otherwise.
    Uses panchang library if available and API matches, otherwise uses a robust mathematical approximation.
    """
    if Location and panchang_calc and hasattr(panchang_calc, 'compute'):
        try:
            delhi = Location(lat=28.6139, lng=77.2090, tz="Asia/Kolkata")
            today_panchang = panchang_calc.compute(date_obj.date(), delhi)
            tithi_name = today_panchang.tithi.name.lower()
            if 'amavasya' in tithi_name or 'purnima' in tithi_name:
                return 1
            return 0
        except Exception:
            pass # Fallback

    # Mathematical fallback (approximation)
    # Jan 6, 2000 was a New Moon (Amavasya)
    new_moon = pd.Timestamp('2000-01-06').date()
    days = (date_obj.date() - new_moon).days
    phase = days % 29.53058867
    
    # Amavasya is around phase 0 (or 29.5)
    # Purnima is around phase 14.76
    if phase < 1 or phase > 28.5:
        return 1 # Amavasya
    elif 13.7 < phase < 15.7:
        return 1 # Purnima
    
    return 0

def create_mock_data(filepath, n_rows=1000):
    print(f"Creating mock dataset at {filepath} for testing...")
    np.random.seed(42)
    start_date = pd.Timestamp('2020-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_rows)]
    
    data = []
    for d in dates:
        # Skip Sundays
        if d.weekday() == 6:
            continue
        row = {
            'Date': d,
            'Day_of_Week': d.strftime('%A'),
        }
        # Cards (0-9)
        m_cards = np.random.choice(10, 3, replace=False)
        e_cards = np.random.choice(10, 3, replace=False)
        
        row['Morning_card1'] = m_cards[0]
        row['Morning_card2'] = m_cards[1]
        row['Morning_card3'] = m_cards[2]
        row['Morning_number'] = sum(m_cards) % 10
        
        row['Evening_number1'] = e_cards[0]
        row['Evening_number2'] = e_cards[1]
        row['Evening_number3'] = e_cards[2]
        row['Evening_number'] = sum(e_cards) % 10
        
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Mock data created with {len(df)} rows.")

def preprocess_and_engineer(df):
    print("Starting preprocessing and feature engineering...")
    # 1. Ensure chronological order
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Continuous Draw_Index
    df['Draw_Index'] = range(1, len(df) + 1)
    
    # 3. Sort the Morning and Evening cards horizontally
    morning_cards = ['Morning_card1', 'Morning_card2', 'Morning_card3']
    evening_cards = ['Evening_number1', 'Evening_number2', 'Evening_number3']
    
    df[morning_cards] = np.sort(df[morning_cards].values, axis=1)
    df[evening_cards] = np.sort(df[evening_cards].values, axis=1)
    
    # 4. Feature: Lags
    for lag in [1, 2, 3]:
        df[f'Morning_lag_{lag}'] = df['Morning_number'].shift(lag)
        df[f'Evening_lag_{lag}'] = df['Evening_number'].shift(lag)
        
    # 5. Feature: Rolling Frequencies of digits in raw cards
    print("Calculating rolling frequencies...")
    all_cards_dummies = pd.DataFrame(0, index=df.index, columns=range(10))
    for col in morning_cards + evening_cards:
        dummies = pd.get_dummies(df[col])
        for val in dummies.columns:
            if val in all_cards_dummies.columns:
                all_cards_dummies[val] += dummies[val]

    for window in [14, 30]:
        rolling_sums = all_cards_dummies.rolling(window=window).sum().shift(1).fillna(0)
        rolling_sums.columns = [f'RollingFreq_{window}_digit_{c}' for c in range(10)]
        df = pd.concat([df, rolling_sums], axis=1)
        
    # 6. Cultural Bias: Holidays & Diwali window
    print("Calculating cultural bias features...")
    in_holidays = holidays.IN(years=df['Date'].dt.year.unique().tolist())
    diwali_dates = [date for date, name in in_holidays.items() if 'Diwali' in name]
    
    def is_festival_window(date):
        if date in in_holidays:
            return 1
        for d_date in diwali_dates:
            if abs((date.date() - d_date).days) <= 3:
                return 1
        return 0

    df['Is_Festival'] = df['Date'].apply(is_festival_window)
    
    # 7. Cultural Bias: Lunar Phase
    df['Is_Amavasya_Purnima'] = df['Date'].apply(get_lunar_phase)
    
    # 8. Post-Holiday Reopening
    df['Days_Since_Last_Draw'] = df['Date'].diff().dt.days.fillna(1)
    
    # Drop rows with NaN (due to lags)
    df = df.dropna().reset_index(drop=True)
    return df

def walk_forward_validation(df):
    print("Starting Walk-Forward Validation Engine...")
    
    # Define features and targets
    exclude_cols = ['Date', 'Day_of_Week', 'Morning_number', 'Evening_number']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features]
    y_m = df['Morning_number'].astype(int)
    y_e = df['Evening_number'].astype(int)
    
    # Model parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': 10,
        'eval_metric': 'mlogloss',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1
    }
    
    model_m = xgb.XGBClassifier(**params)
    model_e = xgb.XGBClassifier(**params)
    
    initial_window = 500
    if len(df) <= initial_window:
        print(f"Dataset too small for walk-forward validation (needs > {initial_window} rows). Training on all available data for prediction.")
        model_m.fit(X, y_m)
        model_e.fit(X, y_e)
        return model_m, model_e, X, y_m, y_e
    
    m_preds = []
    e_preds = []
    m_actuals = []
    e_actuals = []
    
    print(f"Total draws for validation: {len(df) - initial_window}")
    
    for i in range(initial_window, len(df)):
        if (i - initial_window) % 50 == 0:
            print(f"Validating step {i - initial_window} / {len(df) - initial_window}")
            
        X_train = X.iloc[:i]
        y_train_m = y_m.iloc[:i]
        y_train_e = y_e.iloc[:i]
        
        X_test = X.iloc[[i]]
        
        # Train
        model_m.fit(X_train, y_train_m)
        model_e.fit(X_train, y_train_e)
        
        # Predict
        m_pred = model_m.predict(X_test)[0]
        e_pred = model_e.predict(X_test)[0]
        
        m_preds.append(m_pred)
        e_preds.append(e_pred)
        m_actuals.append(y_m.iloc[i])
        e_actuals.append(y_e.iloc[i])
        
    m_acc = accuracy_score(m_actuals, m_preds)
    e_acc = accuracy_score(e_actuals, e_preds)
    
    print(f"Walk-Forward Morning Accuracy:  {m_acc*100:.2f}% (Baseline is 10%)")
    print(f"Walk-Forward Evening Accuracy:  {e_acc*100:.2f}% (Baseline is 10%)")
    
    # Train final models on all data for tomorrow's prediction
    model_m.fit(X, y_m)
    model_e.fit(X, y_e)
    
    return model_m, model_e, X, y_m, y_e

def predict_tomorrow(model_m, model_e, df, X):
    today_date = df['Date'].iloc[-1]
    tomorrow_date = today_date + pd.Timedelta(days=1)
    date_str = tomorrow_date.strftime('%Y-%m-%d')
    day_str = tomorrow_date.strftime('%A')
    print(f"\n--- Generating Prediction for Tomorrow: {date_str} ({day_str}) ---")
    # For predicting tomorrow, we take the last row and assume it represents the most recent state
    # In a real scenario, you'd append tomorrow's date and shift features properly.
    # Here, we will just use the last available feature set as a placeholder for tomorrow's feature state
    # if tomorrow's lags are based on today's actuals.
    
    # The last row in df is 'today'. Tomorrow's features would use 'today' as lag_1.
    # Let's construct a synthetic 'tomorrow' row:
    tomorrow_row = X.iloc[[-1]].copy()
    tomorrow_row['Draw_Index'] += 1
    
    # Shift lags
    for lag in [3, 2]:
        tomorrow_row[f'Morning_lag_{lag}'] = tomorrow_row[f'Morning_lag_{lag-1}']
        tomorrow_row[f'Evening_lag_{lag}'] = tomorrow_row[f'Evening_lag_{lag-1}']
    
    tomorrow_row['Morning_lag_1'] = df['Morning_number'].iloc[-1]
    tomorrow_row['Evening_lag_1'] = df['Evening_number'].iloc[-1]
    
    # We leave rolling frequencies as they are from the end of today (approximate)
    
    # Predict probabilities
    m_probs = model_m.predict_proba(tomorrow_row)[0]
    e_probs = model_e.predict_proba(tomorrow_row)[0]
    
    print("Morning Draw Softmax Probabilities:")
    for i, p in enumerate(m_probs):
        print(f"  Number {i}: {p*100:.1f}%")
        
    print("\nEvening Draw Softmax Probabilities:")
    for i, p in enumerate(e_probs):
        print(f"  Number {i}: {p*100:.1f}%")

def load_and_clean_data(filepath):
    print(f"Loading and enhancing data from {filepath}...")
    df = pd.read_csv(filepath)
    
    if 'Morning_Card1' in df.columns:
        # Drop rows with '*' indicating no draw
        df = df[df['Morning_Card1'] != '*'].copy()
        
        # Rename columns to match pipeline logic
        df = df.rename(columns={
            'Morning_Card1': 'Morning_card1',
            'Morning_Card2': 'Morning_card2',
            'Morning_Card3': 'Morning_card3',
            'Evening_Card1': 'Evening_number1',
            'Evening_Card2': 'Evening_number2',
            'Evening_Card3': 'Evening_number3'
        })
        
        # Convert cards to integers
        card_cols = ['Morning_card1', 'Morning_card2', 'Morning_card3', 
                     'Evening_number1', 'Evening_number2', 'Evening_number3']
        for col in card_cols:
            df[col] = df[col].astype(int)
            
        # Re-calculate the actual outcome numbers (Modulo 10) instead of relying on Jodi_Number string
        df['Morning_number'] = (df['Morning_card1'] + df['Morning_card2'] + df['Morning_card3']) % 10
        df['Evening_number'] = (df['Evening_number1'] + df['Evening_number2'] + df['Evening_number3']) % 10

    # Ensure chronological order
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    if os.path.exists('combined_data.csv'):
        dataset_path = 'combined_data.csv'
    elif os.path.exists('draw_data.csv'):
        dataset_path = 'draw_data.csv'
    else:
        dataset_path = 'dataset.csv'
    
    if not os.path.exists(dataset_path):
        create_mock_data(dataset_path, n_rows=600)
        
    df = load_and_clean_data(dataset_path)
    
    processed_df = preprocess_and_engineer(df)
    model_m, model_e, X, y_m, y_e = walk_forward_validation(processed_df)
    predict_tomorrow(model_m, model_e, processed_df, X)
    print("\nPipeline execution complete.")
