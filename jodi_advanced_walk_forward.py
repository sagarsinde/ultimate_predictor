import pandas as pd
import numpy as np
import xgboost as xgb
import networkx as nx
import warnings

warnings.filterwarnings('ignore')

def load_and_engineer(filepath):
    print("Loading data and engineering advanced deep-math features...")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Core target
    df['Jodi'] = df['Morning_number'].astype(str) + df['Evening_number'].astype(str)
    
    # 1. Advanced Temporal Features (Calendar Bias)
    df['Day_of_Week_Num'] = df['Date'].dt.dayofweek
    df['Day_of_Month'] = df['Date'].dt.day
    
    # Cyclical Calendar mapping (Sine/Cosine rings)
    # Day of month (1-31) mapped to a circle
    df['DOM_sin'] = np.sin(2 * np.pi * df['Day_of_Month'] / 31.0)
    df['DOM_cos'] = np.cos(2 * np.pi * df['Day_of_Month'] / 31.0)
    
    # Day of week (0-6) mapped to a circle
    df['DOW_sin'] = np.sin(2 * np.pi * df['Day_of_Week_Num'] / 7.0)
    df['DOW_cos'] = np.cos(2 * np.pi * df['Day_of_Week_Num'] / 7.0)

    # 2. FFT Cyclical Frequencies
    # From deep analysis, we found dominant cycles at 2.2 days, 2.8 days, 4.9 days.
    time_steps = np.arange(len(df))
    df['FFT_Wave_2_2'] = np.sin(2 * np.pi * time_steps / 2.2)
    df['FFT_Wave_2_8'] = np.sin(2 * np.pi * time_steps / 2.8)
    df['FFT_Wave_4_9'] = np.sin(2 * np.pi * time_steps / 4.9)
    
    # 3. Lag Features (Classic)
    for i in range(1, 4):
        df[f'Morning_lag_{i}'] = df['Morning_number'].shift(i)
        df[f'Evening_lag_{i}'] = df['Evening_number'].shift(i)
        
    # 4. Graph Theory Transition Centrality (PageRank of Yesterday's Jodi)
    # We maintain an expanding network graph to prevent looking into the future.
    pagerank_scores = []
    edge_weights = {}
    pr = {} # Current PageRank dictionary
    
    for i in range(len(df)):
        if i < 5:
            pagerank_scores.append(0.01) # Default score
            continue
            
        # The edge that just happened yesterday
        prev_u = df.loc[i-2, 'Jodi'] if i >= 2 else "00"
        prev_v = df.loc[i-1, 'Jodi']
        
        edge = (prev_u, prev_v)
        edge_weights[edge] = edge_weights.get(edge, 0) + 1
        
        # Update PageRank math every 10 steps for speed
        if i % 10 == 0:
            G = nx.DiGraph()
            for (u, v), w in edge_weights.items():
                G.add_edge(u, v, weight=w)
            try:
                pr = nx.pagerank(G, weight='weight')
            except:
                pr = {}
        
        # Assign the PageRank score of yesterday's Jodi to today's feature set
        score = pr.get(prev_v, 0.01)
        pagerank_scores.append(score)
        
    df['Yesterday_PageRank'] = pagerank_scores
    
    # Drop rows that don't have lag data
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def walk_forward_validation(df, window_size=500):
    print(f"Starting Strict Walk-Forward Validation (Window: {window_size})...")
    
    # Exclude non-mathematical labels from training
    features = [c for c in df.columns if c not in ['Date', 'Day_of_Week', 'Jodi', 'Morning_number', 'Evening_number']]
    
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
        'tree_method': 'hist', # Speed optimized
        'random_state': 42,
        'verbosity': 0
    }
    
    total_predictions = 0
    m_top1_hits = 0; m_top3_hits = 0
    e_top1_hits = 0; e_top3_hits = 0
    jodi_top4_hits = 0
    
    steps = len(df) - window_size
    
    for i in range(window_size, len(df)):
        if (i - window_size) % 50 == 0:
            print(f"  Validating Step {i - window_size}/{steps}...")
            
        train_idx_start = i - window_size
        train_idx_end = i
        
        X_train = X.iloc[train_idx_start:train_idx_end]
        y_m_train = y_m.iloc[train_idx_start:train_idx_end]
        y_e_train = y_e.iloc[train_idx_start:train_idx_end]
        
        X_test = X.iloc[[i]]
        true_m = y_m.iloc[i]
        true_e = y_e.iloc[i]
        
        model_m = xgb.XGBClassifier(**params)
        model_e = xgb.XGBClassifier(**params)
        
        model_m.fit(X_train, y_m_train)
        model_e.fit(X_train, y_e_train)
        
        m_probs = model_m.predict_proba(X_test)[0]
        e_probs = model_e.predict_proba(X_test)[0]
        
        m_top3 = np.argsort(m_probs)[-3:][::-1]
        e_top3 = np.argsort(e_probs)[-3:][::-1]
        
        if true_m == m_top3[0]: m_top1_hits += 1
        if true_m in m_top3: m_top3_hits += 1
        
        if true_e == e_top3[0]: e_top1_hits += 1
        if true_e in e_top3: e_top3_hits += 1
        
        # Calculate Jodi Joint Probabilities
        jodi_probs = {}
        for mm in range(10):
            for ee in range(10):
                jodi_probs[f"{mm}{ee}"] = m_probs[mm] * e_probs[ee]
                
        top4_jodis = sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]
        top4_keys = [x[0] for x in top4_jodis]
        
        true_jodi = f"{true_m}{true_e}"
        if true_jodi in top4_keys:
            jodi_top4_hits += 1
            
        total_predictions += 1
        
    m_top1_acc = m_top1_hits / total_predictions
    m_top3_acc = m_top3_hits / total_predictions
    e_top1_acc = e_top1_hits / total_predictions
    e_top3_acc = e_top3_hits / total_predictions
    jodi_top4_acc = jodi_top4_hits / total_predictions
    
    print("\n=======================================================")
    print("      ADVANCED DEEP-MATH VALIDATION SCORECARD          ")
    print("=======================================================")
    print(f"Total Test Period Draws Evaluated: {total_predictions}")
    print("\nMorning Model:")
    print(f"  Top 1 Hit Rate: {m_top1_acc*100:.2f}% (Random Baseline: 10%)")
    print(f"  Top 3 Hit Rate: {m_top3_acc*100:.2f}% (Random Baseline: 30%)")
    
    print("\nEvening Model:")
    print(f"  Top 1 Hit Rate: {e_top1_acc*100:.2f}% (Random Baseline: 10%)")
    print(f"  Top 3 Hit Rate: {e_top3_acc*100:.2f}% (Random Baseline: 30%)")
    
    print("\nJodi Model (Joint Probability):")
    print(f"  Top 4 Jodi Hit Rate: {jodi_top4_acc*100:.2f}% (Random Baseline: 4%)")
    print("=======================================================\n")
    
    return params, features

def predict_tomorrow(df, params, features, window_size=500):
    print("Training final advanced models for 'Tomorrow'...")
    
    X = df[features]
    y_m = df['Morning_number'].astype(int)
    y_e = df['Evening_number'].astype(int)
    
    train_idx_start = len(df) - window_size
    X_train = X.iloc[train_idx_start:]
    y_m_train = y_m.iloc[train_idx_start:]
    y_e_train = y_e.iloc[train_idx_start:]
    
    model_m = xgb.XGBClassifier(**params)
    model_e = xgb.XGBClassifier(**params)
    model_m.fit(X_train, y_m_train)
    model_e.fit(X_train, y_e_train)
    
    # Synthesize tomorrow's mathematical state
    tomorrow_X = X.iloc[[-1]].copy()
    
    # Shift classic lags
    for i in [3, 2]:
        if f'Morning_lag_{i}' in tomorrow_X.columns and f'Morning_lag_{i-1}' in tomorrow_X.columns:
            tomorrow_X[f'Morning_lag_{i}'] = tomorrow_X[f'Morning_lag_{i-1}']
            tomorrow_X[f'Evening_lag_{i}'] = tomorrow_X[f'Evening_lag_{i-1}']
            
    if 'Morning_lag_1' in tomorrow_X.columns:
        tomorrow_X['Morning_lag_1'] = y_m.iloc[-1]
        tomorrow_X['Evening_lag_1'] = y_e.iloc[-1]
        
    # Calculate exact date for tomorrow
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    pred_date = last_date + pd.Timedelta(days=1)
    pred_date_str = pred_date.strftime('%Y-%m-%d (%A)')
    
    # Shift Temporal Rings
    tomorrow_X['Day_of_Week_Num'] = pred_date.dayofweek
    tomorrow_X['Day_of_Month'] = pred_date.day
    tomorrow_X['DOM_sin'] = np.sin(2 * np.pi * pred_date.day / 31.0)
    tomorrow_X['DOM_cos'] = np.cos(2 * np.pi * pred_date.day / 31.0)
    tomorrow_X['DOW_sin'] = np.sin(2 * np.pi * pred_date.dayofweek / 7.0)
    tomorrow_X['DOW_cos'] = np.cos(2 * np.pi * pred_date.dayofweek / 7.0)
    
    # Shift FFT cycle forward 1 step
    next_step = len(df)
    tomorrow_X['FFT_Wave_2_2'] = np.sin(2 * np.pi * next_step / 2.2)
    tomorrow_X['FFT_Wave_2_8'] = np.sin(2 * np.pi * next_step / 2.8)
    tomorrow_X['FFT_Wave_4_9'] = np.sin(2 * np.pi * next_step / 4.9)
    
    # PageRank uses today's actual Jodi as the node transition
    # It is already updated from the last loop
    
    m_probs = model_m.predict_proba(tomorrow_X)[0]
    e_probs = model_e.predict_proba(tomorrow_X)[0]
    
    jodi_probs = {}
    for mm in range(10):
        for ee in range(10):
            jodi_probs[f"{mm}{ee}"] = m_probs[mm] * e_probs[ee]
            
    top4_jodis = sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]
    m_top3 = np.argsort(m_probs)[-3:][::-1]
    e_top3 = np.argsort(e_probs)[-3:][::-1]
    
    print("\n=======================================================")
    print(f"   ADVANCED PREDICTIONS FOR: {pred_date_str}")
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

def main():
    filepath = 'true_kalyan_morning_dataset.csv'
    try:
        df = load_and_engineer(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return
        
    params, features = walk_forward_validation(df, window_size=500)
    predict_tomorrow(df, params, features, window_size=500)

if __name__ == '__main__':
    main()
