import pandas as pd
import numpy as np
import networkx as nx

def generate_features(df_input):
    """
    Core Feature Engineering Module (Phase 1 & 3 Shared Logic)
    Strictly causal. No future lookahead. 
    Generates the mathematical state of the universe strictly prior to the draw.
    """
    print("Generating strictly causal features (Lags, Streaks, Gamblers Fallacy, PageRank)...")
    df = df_input.copy()
    
    # 1. Cyclical Time Features (Day of Week bias)
    # The new KALYAN dataset is Mon-Sat. Let's map Mon=0 to Sat=5.
    dow_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5}
    if 'Day_of_Week' in df.columns:
        df['DOW_Num'] = df['Day_of_Week'].map(dow_map).fillna(0)
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        # Day of Week Encoding (Sine/Cosine for cyclical nature)
        df['DOW_Num'] = df['Date'].dt.dayofweek
        
    df['DOW_sin'] = np.sin(2 * np.pi * df['DOW_Num'] / 5.0)
    df['DOW_cos'] = np.cos(2 * np.pi * df['DOW_Num'] / 5.0)
    
    # 2. Classic Lags (1 to 7 Days)
    for i in range(1, 8):
        df[f'Morning_lag_{i}'] = df['Morning_number'].shift(i)
        df[f'Evening_lag_{i}'] = df['Evening_number'].shift(i)
        
    # Initialize complex causal features
    days_since_hit_m = {str(d): 0 for d in range(10)}
    days_since_hit_e = {str(d): 0 for d in range(10)}
    
    # Graph for PageRank
    edge_weights = {}
    
    # Store calculated features to append later
    causal_features = []
    
    # Causal loop over history
    for i in range(len(df)):
        row_features = {}
        
        # ---------------------------------------------------------
        # A. Gambler's Fallacy (Days Since Last Hit)
        # Represents the psychological pressure of the crowd
        # ---------------------------------------------------------
        for d in range(10):
            row_features[f'M_Days_Since_{d}'] = days_since_hit_m[str(d)]
            row_features[f'E_Days_Since_{d}'] = days_since_hit_e[str(d)]
            
        # ---------------------------------------------------------
        # B. Anti-Crowd Streaks (Hits in last 7 days)
        # ---------------------------------------------------------
        if i >= 7:
            # We look at the window [i-7 : i-1]
            past_7_m = df.loc[i-7:i-1, 'Morning_number'].astype(str).tolist()
            past_7_e = df.loc[i-7:i-1, 'Evening_number'].astype(str).tolist()
            
            for d in range(10):
                row_features[f'M_Hits_Last_7_{d}'] = past_7_m.count(str(d))
                row_features[f'E_Hits_Last_7_{d}'] = past_7_e.count(str(d))
        else:
            for d in range(10):
                row_features[f'M_Hits_Last_7_{d}'] = 0
                row_features[f'E_Hits_Last_7_{d}'] = 0
                
        # ---------------------------------------------------------
        # C. Graph Theory (Expanding PageRank)
        # ---------------------------------------------------------
        if i >= 2:
            prev_u = f"{df.loc[i-2, 'Morning_number']}{df.loc[i-2, 'Evening_number']}"
            prev_v = f"{df.loc[i-1, 'Morning_number']}{df.loc[i-1, 'Evening_number']}"
            edge = (prev_u, prev_v)
            edge_weights[edge] = edge_weights.get(edge, 0) + 1
            
        pr_score = 0.01 # Default
        if i % 5 == 0 and len(edge_weights) > 0: # Update graph every 5 steps for speed
            G = nx.DiGraph()
            for (u, v), w in edge_weights.items():
                G.add_edge(u, v, weight=w)
            try:
                # We need to save the pr_dict to use for the next 5 steps
                pr_dict = nx.pagerank(G, weight='weight')
            except:
                pr_dict = {}
                
        if i >= 1:
            yesterday_jodi = f"{df.loc[i-1, 'Morning_number']}{df.loc[i-1, 'Evening_number']}"
            pr_score = pr_dict.get(yesterday_jodi, 0.01) if 'pr_dict' in locals() else 0.01
            
        row_features['Yesterday_PageRank'] = pr_score
        
        # ---------------------------------------------------------
        # D. FFT Deterministic Math Waves (Cycles)
        # ---------------------------------------------------------
        row_features['FFT_Wave_2_2'] = np.sin(2 * np.pi * i / 2.2)
        row_features['FFT_Wave_4_9'] = np.sin(2 * np.pi * i / 4.9)
        
        causal_features.append(row_features)
        
        # ---------------------------------------------------------
        # UPDATE STATE FOR NEXT ROW (Strictly Causal)
        # ---------------------------------------------------------
        m_val = str(df.loc[i, 'Morning_number'])
        e_val = str(df.loc[i, 'Evening_number'])
        
        for d in range(10):
            if str(d) == m_val:
                days_since_hit_m[str(d)] = 0
            else:
                days_since_hit_m[str(d)] += 1
                
            if str(d) == e_val:
                days_since_hit_e[str(d)] = 0
            else:
                days_since_hit_e[str(d)] += 1

    # Merge features back to dataframe
    feature_df = pd.DataFrame(causal_features)
    df = pd.concat([df, feature_df], axis=1)
    
    # Drop rows with NaNs (first 7 rows due to lags)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
