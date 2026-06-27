import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

def load_and_engineer_features(filepath='draw_data.csv'):
    print("Loading data and engineering advanced features...")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate Morning and Evening Numbers from raw cards
    for prefix in ['Morning', 'Evening']:
        for i in [1, 2, 3]:
            col = f'{prefix}_Card{i}'
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        df[f'{prefix}_Sum'] = df[f'{prefix}_Card1'] + df[f'{prefix}_Card2'] + df[f'{prefix}_Card3']
        df[f'{prefix}_Number'] = df[f'{prefix}_Sum'] % 10
    
    # 1. Target Variables
    df['Morning_Target'] = df['Morning_Number'].astype(int)
    df['Evening_Target'] = df['Evening_Number'].astype(int)
    
    # 2. Cyclical Time Encoding (Day of Week)
    # Mapping Mon=0, Sun=6
    day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    df['Day_Num'] = df['Day_of_Week'].map(day_map)
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day_Num'] / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day_Num'] / 7)
    
    # 3. Lag Features (Memory of previous draws)
    for lag in [1, 2, 3, 5]:
        df[f'Morning_Lag_{lag}'] = df['Morning_Target'].shift(lag)
        df[f'Evening_Lag_{lag}'] = df['Evening_Target'].shift(lag)
        
    # 4. Rolling Statistics (Hot Streaks)
    # Let's track rolling sum and rolling std dev of the numbers (to see if they are clustering high or low)
    df['Morning_RollMean_7'] = df['Morning_Target'].shift(1).rolling(window=7).mean()
    df['Morning_RollStd_7']  = df['Morning_Target'].shift(1).rolling(window=7).std()
    
    # Drop rows with NaNs caused by lagging
    df = df.dropna().reset_index(drop=True)
    print(f"Dataset ready. Total viable records for training/testing: {len(df)}")
    return df

def build_markov_chain(df, target_col='Morning_Target'):
    """
    Builds a 1st order Markov Transition Matrix.
    Calculates the exact probability of moving from State A (Yesterday) to State B (Today).
    """
    print(f"\n--- Markov Chain Transition Matrix ({target_col}) ---")
    transitions = defaultdict(lambda: defaultdict(int))
    
    for i in range(1, len(df)):
        prev_state = df[target_col].iloc[i-1]
        curr_state = df[target_col].iloc[i]
        transitions[prev_state][curr_state] += 1
        
    # Convert counts to probabilities
    prob_matrix = np.zeros((10, 10))
    for i in range(10):
        total_transitions = sum(transitions[i].values())
        if total_transitions > 0:
            for j in range(10):
                prob_matrix[i, j] = transitions[i][j] / total_transitions
                
    # Plotting the Transition Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_matrix, annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': 'Transition Probability'})
    plt.title(f'Markov Transition Probabilities - {target_col}\n(Y-axis: Yesterday, X-axis: Today)')
    plt.ylabel('Yesterday\'s Number')
    plt.xlabel('Today\'s Number')
    plt.tight_layout()
    plt.savefig(f'markov_matrix_{target_col.lower()}.png')
    plt.close()
    print(f"Saved Markov Transition Matrix visualization to markov_matrix_{target_col.lower()}.png")
    
    return prob_matrix

def walk_forward_validation(df, features, target_col):
    """
    Simulates real-world predicting using chronological walk-forward validation.
    We train on past data and predict the strict future.
    """
    print(f"\n--- XGBoost Walk-Forward Validation ({target_col}) ---")
    
    # We will use the last 100 days for strict testing
    test_size = 100
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    X_train = train_df[features]
    y_train = train_df[target_col]
    
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    # XGBoost Classifier setup
    # objective multi:softmax since we are predicting exactly 1 of 10 classes (0-9)
    model = xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=10,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    
    print("Training XGBoost Classifier...")
    model.fit(X_train, y_train)
    
    print("Predicting the last 100 days...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nWalk-Forward Accuracy: {accuracy * 100:.2f}%")
    print(f"Random Guess Baseline: 10.00%")
    
    if accuracy > 0.15:
        print(">>> MODEL DETECTED AN EDGE! Predicting significantly above baseline probability.")
    else:
        print(">>> No significant predictive edge found. The draw process appears mechanically sound.")
        
    return model, accuracy

if __name__ == "__main__":
    df = load_and_engineer_features('draw_data.csv')
    
    # Build Markov Chains
    morning_markov = build_markov_chain(df, 'Morning_Target')
    evening_markov = build_markov_chain(df, 'Evening_Target')
    
    # Define XGBoost Features
    feature_cols = [
        'Day_Sin', 'Day_Cos', 
        'Morning_Lag_1', 'Morning_Lag_2', 'Morning_Lag_3', 'Morning_Lag_5',
        'Evening_Lag_1', 'Evening_Lag_2', 'Evening_Lag_3', 'Evening_Lag_5',
        'Morning_RollMean_7', 'Morning_RollStd_7'
    ]
    
    # Walk Forward Validation
    model_morning, acc_morning = walk_forward_validation(df, feature_cols, 'Morning_Target')
    model_evening, acc_evening = walk_forward_validation(df, feature_cols, 'Evening_Target')
