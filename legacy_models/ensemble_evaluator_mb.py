import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# 1. Define LSTM Architecture (Must match exactly)
class SattaLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2, num_classes=10):
        super(SattaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc_m = nn.Linear(hidden_size, num_classes)
        self.fc_e = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        return self.fc_m(last_out), self.fc_e(last_out)

def main():
    print("--- Phase 8: Hybrid Ensemble Engine (Main Bazar) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load the Brains (XGBoost + LSTM)
    try:
        # Load XGBoost Morning Model
        xgb_m = joblib.load('mb_morning_model.joblib')
        with open('mb_model_features.json', 'r') as f:
            xgb_features = json.load(f)
            
        # Load LSTM Evening Model
        lstm_model = SattaLSTM().to(device)
        lstm_model.load_state_dict(torch.load('mb_lstm_model.pth', map_location=device, weights_only=True))
        lstm_model.eval()
    except FileNotFoundError as e:
        print(f"Error: Models missing. {e}")
        return
        
    print("Opening the Data Lake...")
    # We must use the Feature Store because XGBoost needs the causal features
    df = pd.read_csv('main_bazar_feature_store.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 3. Unlock the 2026 Vault
    vault_mask = df['Date'] >= '2026-01-01'
    vault_indices = df[vault_mask].index.tolist()
    
    print(f"2026 Vault Unlocked. Testing Hybrid Ensemble on {len(vault_indices)} unseen draws...\n")
    
    # Prepare LSTM Sequence Data
    raw_m = df['Morning_number'].astype(int).values
    raw_e = df['Evening_number'].astype(int).values
    
    encoder = OneHotEncoder(sparse_output=False, categories=[range(10)])
    m_encoded = encoder.fit_transform(raw_m.reshape(-1, 1))
    e_encoded = encoder.fit_transform(raw_e.reshape(-1, 1))
    daily_vectors = np.hstack([m_encoded, e_encoded])
    
    # Tracking Scorecard
    m_hits_1, m_hits_3 = 0, 0
    e_hits_1, e_hits_3 = 0, 0
    jodi_hits = 0
    
    # 4. Simulating the Timeline
    for idx in vault_indices:
        if idx < 30:
            continue
            
        actual_m = raw_m[idx]
        actual_e = raw_e[idx]
        actual_jodi = f"{actual_m}{actual_e}"
        
        # --- A. XGBoost Predicts Morning ---
        row = df.iloc[idx]
        X_test = row[xgb_features].to_frame().T.astype(float)
        m_probs = xgb_m.predict_proba(X_test)[0]
        
        # --- B. LSTM Predicts Evening ---
        sequence = daily_vectors[idx-30 : idx]
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, out_e = lstm_model(seq_tensor)
            e_probs = torch.nn.functional.softmax(out_e, dim=1).cpu().numpy()[0]
            
        # --- C. Calculate Ensemble ---
        m_top3 = np.argsort(m_probs)[-3:][::-1].tolist()
        e_top3 = np.argsort(e_probs)[-3:][::-1].tolist()
        
        # Joint Jodi Probability (Multiply XGBoost by LSTM)
        jodi_probs = {}
        for mm in range(10):
            for ee in range(10):
                jodi_probs[f"{mm}{ee}"] = m_probs[mm] * e_probs[ee]
                
        top4_jodis = [k for k, v in sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]]
        
        # Score
        if actual_m == m_top3[0]: m_hits_1 += 1
        if actual_m in m_top3: m_hits_3 += 1
        
        if actual_e == e_top3[0]: e_hits_1 += 1
        if actual_e in e_top3: e_hits_3 += 1
        
        if actual_jodi in top4_jodis: jodi_hits += 1
        
    # 5. Print Final Scorecard
    total_draws = len(vault_indices)
    print("=======================================================")
    print("      HYBRID ENSEMBLE SCORECARD (2026 VAULT)")
    print("=======================================================")
    print(f"Total Test Period Draws Evaluated: {total_draws}\n")
    
    print("--- XGBOOST MORNING MODEL ---")
    print(f"Top 1 Hit Rate: {(m_hits_1/total_draws)*100:.2f}%  (Random Baseline: 10%)")
    print(f"Top 3 Hit Rate: {(m_hits_3/total_draws)*100:.2f}%  (Random Baseline: 30%)\n")
    
    print("--- LSTM EVENING MODEL ---")
    print(f"Top 1 Hit Rate: {(e_hits_1/total_draws)*100:.2f}%  (Random Baseline: 10%)")
    print(f"Top 3 Hit Rate: {(e_hits_3/total_draws)*100:.2f}%  (Random Baseline: 30%)\n")
    
    print("--- ENSEMBLE JODI JOINT PROBABILITY ---")
    print(f"Top 4 Jodi Hit Rate: {(jodi_hits/total_draws)*100:.2f}%  (Random Baseline: 4%)")
    print("=======================================================")

if __name__ == '__main__':
    main()
