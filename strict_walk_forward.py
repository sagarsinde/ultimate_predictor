import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

from pipeline import load_and_clean_data, preprocess_and_engineer

def load_data(filepath='combined_data.csv'):
    """Loads the dataset, engineers features, and separates features and targets."""
    print(f"Loading and engineering data from {filepath}...")
    try:
        raw_df = load_and_clean_data(filepath)
        df = preprocess_and_engineer(raw_df)
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        raise

    # Exclude non-feature columns
    exclude_cols = ['Date', 'Day_of_Week', 'Morning_number', 'Evening_number', 'Draw_Index']
    features = [col for col in df.columns if col not in exclude_cols]

    y_m = df['Morning_number'].astype(int)
    y_e = df['Evening_number'].astype(int)
    X = df[features]
    
    return df, X, y_m, y_e

def walk_forward_validation(df, X, y_m, y_e, window_size=500):
    """
    Executes strict chronological walk-forward validation.
    Trains on a rolling window of PREVIOUS 'window_size' draws,
    predicts the VERY NEXT draw.
    """
    if len(df) <= window_size:
        print(f"Dataset is too small (<= {window_size} rows) for Walk-Forward Validation.")
        return None

    # XGBoost Parameters
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

    # Tracking metrics
    m_top1_hits = 0
    m_top3_hits = 0
    e_top1_hits = 0
    e_top3_hits = 0

    total_predictions = 0

    print("\n=======================================================")
    print("        STARTING STRICT WALK-FORWARD VALIDATION        ")
    print("=======================================================")
    print(f"Rolling window size: {window_size} draws")
    print(f"Total draws to predict: {len(df) - window_size}\n")

    for i in range(window_size, len(df)):
        # 1. Train on previous 'window_size' draws (e.g., Draw 1 to 500)
        start_idx = i - window_size
        end_idx = i

        X_train = X.iloc[start_idx:end_idx]
        y_m_train = y_m.iloc[start_idx:end_idx]
        y_e_train = y_e.iloc[start_idx:end_idx]

        # 2. Predict the VERY NEXT unseen draw (e.g., Draw 501)
        X_test = X.iloc[[i]]
        y_m_test = y_m.iloc[i]
        y_e_test = y_e.iloc[i]

        model_m = xgb.XGBClassifier(**params)
        model_e = xgb.XGBClassifier(**params)

        model_m.fit(X_train, y_m_train)
        model_e.fit(X_train, y_e_train)

        # 3. Extract Softmax Probabilities
        m_probs = model_m.predict_proba(X_test)[0]
        e_probs = model_e.predict_proba(X_test)[0]

        # Find Top 3 predicted digits
        m_top3_pred = np.argsort(m_probs)[-3:][::-1]
        e_top3_pred = np.argsort(e_probs)[-3:][::-1]

        m_top1_pred = m_top3_pred[0]
        e_top1_pred = e_top3_pred[0]

        # 4. Compare and log result
        total_predictions += 1
        
        if m_top1_pred == y_m_test:
            m_top1_hits += 1
        if y_m_test in m_top3_pred:
            m_top3_hits += 1

        if e_top1_pred == y_e_test:
            e_top1_hits += 1
        if y_e_test in e_top3_pred:
            e_top3_hits += 1
        
        if (i - window_size + 1) % 50 == 0:
            print(f"Processed {i - window_size + 1} / {len(df) - window_size} predictions...")

    # Calculate blind hit rate
    m_top1_acc = m_top1_hits / total_predictions
    m_top3_acc = m_top3_hits / total_predictions
    e_top1_acc = e_top1_hits / total_predictions
    e_top3_acc = e_top3_hits / total_predictions

    print("\n=======================================================")
    print("                 VALIDATION SCORECARD                  ")
    print("=======================================================")
    print(f"Total Test Period Draws Evaluated: {total_predictions}")
    print("\nMorning Model:")
    print(f"  Top 1 Hit Rate (Accuracy): {m_top1_acc*100:.2f}% (Random Baseline 10%)")
    print(f"  Top 3 Hit Rate:            {m_top3_acc*100:.2f}% (Random Baseline 30%)")
    
    print("\nEvening Model:")
    print(f"  Top 1 Hit Rate (Accuracy): {e_top1_acc*100:.2f}% (Random Baseline 10%)")
    print(f"  Top 3 Hit Rate:            {e_top3_acc*100:.2f}% (Random Baseline 30%)")
    print("=======================================================\n")
    
    return params

def predict_tomorrow(df, X, y_m, y_e, params, window_size=500):
    """
    Trains one last time on the final 500 rows and predicts 'Tomorrow'.
    """
    print(f"Training final models on the last {window_size} rows for 'Tomorrow'...")
    final_start_idx = len(df) - window_size
    
    X_train_final = X.iloc[final_start_idx:]
    y_m_train_final = y_m.iloc[final_start_idx:]
    y_e_train_final = y_e.iloc[final_start_idx:]

    model_m_final = xgb.XGBClassifier(**params)
    model_e_final = xgb.XGBClassifier(**params)

    model_m_final.fit(X_train_final, y_m_train_final)
    model_e_final.fit(X_train_final, y_e_train_final)
    
    # Synthesize tomorrow's features
    tomorrow_X = X.iloc[[-1]].copy()
    
    # Shift lags for tomorrow if they exist in the feature set
    for lag in [5, 4, 3, 2]:
        if f'Morning_lag_{lag}' in tomorrow_X.columns and f'Morning_lag_{lag-1}' in tomorrow_X.columns:
            tomorrow_X[f'Morning_lag_{lag}'] = tomorrow_X[f'Morning_lag_{lag-1}']
            tomorrow_X[f'Evening_lag_{lag}'] = tomorrow_X[f'Evening_lag_{lag-1}']
            
    if 'Morning_lag_1' in tomorrow_X.columns:
        tomorrow_X['Morning_lag_1'] = y_m.iloc[-1]
        tomorrow_X['Evening_lag_1'] = y_e.iloc[-1]
        
    m_probs_tomorrow = model_m_final.predict_proba(tomorrow_X)[0]
    e_probs_tomorrow = model_e_final.predict_proba(tomorrow_X)[0]

    m_top3_tomorrow = np.argsort(m_probs_tomorrow)[-3:][::-1]
    e_top3_tomorrow = np.argsort(e_probs_tomorrow)[-3:][::-1]

    print("\n=======================================================")
    print("             PREDICTIONS FOR TOMORROW                  ")
    print("=======================================================")
    print("Morning Draw - Exact Top 3 Softmax Probabilities:")
    for num in m_top3_tomorrow:
        print(f"  Digit {num}: {m_probs_tomorrow[num]*100:.2f}%")

    print("\nEvening Draw - Exact Top 3 Softmax Probabilities:")
    for num in e_top3_tomorrow:
        print(f"  Digit {num}: {e_probs_tomorrow[num]*100:.2f}%")
    print("=======================================================\n")


def main():
    filepath = 'combined_data.csv'
    
    try:
        df, X, y_m, y_e = load_data(filepath)
    except Exception as e:
        return

    params = walk_forward_validation(df, X, y_m, y_e, window_size=500)
    
    if params is not None:
        predict_tomorrow(df, X, y_m, y_e, params, window_size=500)

if __name__ == '__main__':
    main()
