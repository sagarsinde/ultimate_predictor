import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# 1. Define LSTM Architecture
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
    print("--- Phase 9: Confidence Threshold Optimizer (Main Bazar) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        xgb_m = joblib.load('mb_morning_model.joblib')
        with open('mb_model_features.json', 'r') as f:
            xgb_features = json.load(f)
            
        lstm_model = SattaLSTM().to(device)
        lstm_model.load_state_dict(torch.load('mb_lstm_model.pth', map_location=device, weights_only=True))
        lstm_model.eval()
    except FileNotFoundError as e:
        print(f"Error: Models missing. {e}")
        return
        
    df = pd.read_csv('main_bazar_feature_store.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    vault_mask = df['Date'] >= '2026-01-01'
    vault_indices = df[vault_mask].index.tolist()
    
    raw_m = df['Morning_number'].astype(int).values
    raw_e = df['Evening_number'].astype(int).values
    
    encoder = OneHotEncoder(sparse_output=False, categories=[range(10)])
    m_encoded = encoder.fit_transform(raw_m.reshape(-1, 1))
    e_encoded = encoder.fit_transform(raw_e.reshape(-1, 1))
    daily_vectors = np.hstack([m_encoded, e_encoded])
    
    # Pre-calculate all predictions to save time
    print("Pre-calculating AI probabilities for 2026 Vault...")
    predictions = []
    for idx in vault_indices:
        if idx < 30: continue
        actual_m = raw_m[idx]
        actual_e = raw_e[idx]
        actual_jodi = f"{actual_m}{actual_e}"
        
        row = df.iloc[idx]
        X_test = row[xgb_features].to_frame().T.astype(float)
        m_probs = xgb_m.predict_proba(X_test)[0]
        
        sequence = daily_vectors[idx-30 : idx]
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, out_e = lstm_model(seq_tensor)
            e_probs = torch.nn.functional.softmax(out_e, dim=1).cpu().numpy()[0]
            
        predictions.append({
            'actual_m': actual_m,
            'actual_e': actual_e,
            'actual_jodi': actual_jodi,
            'm_probs': m_probs,
            'e_probs': e_probs
        })
        
    print(f"Testing Confidence Thresholds from 10% to 30%...\n")
    print(f"{'Threshold':<12} | {'Days Played':<12} | {'Total Inv':<10} | {'Total Won':<10} | {'Net Profit':<10} | {'ROI':<10}")
    print("-" * 75)
    
    best_roi = -100
    best_thresh = 0
    
    # Test thresholds from 10% (0.10) to 30% (0.30)
    for threshold_int in range(10, 31):
        threshold = threshold_int / 100.0
        
        total_invested = 0
        total_won = 0
        days_played = 0
        
        for pred in predictions:
            m_probs = pred['m_probs']
            e_probs = pred['e_probs']
            
            max_m_conf = np.max(m_probs)
            max_e_conf = np.max(e_probs)
            
            play_morning = max_m_conf >= threshold
            play_evening = max_e_conf >= threshold
            play_jodi = play_morning and play_evening
            
            if not play_morning and not play_evening:
                continue # Skip this day completely
                
            days_played += 1
            
            daily_cost = 0
            daily_revenue = 0
            
            m_top3 = np.argsort(m_probs)[-3:][::-1].tolist()
            e_top3 = np.argsort(e_probs)[-3:][::-1].tolist()
            
            if play_morning:
                daily_cost += 150
                if pred['actual_m'] in m_top3:
                    daily_revenue += 450
                    
            if play_evening:
                daily_cost += 150
                if pred['actual_e'] in e_top3:
                    daily_revenue += 450
                    
            if play_jodi:
                daily_cost += 200
                jodi_probs = {}
                for mm in range(10):
                    for ee in range(10):
                        jodi_probs[f"{mm}{ee}"] = m_probs[mm] * e_probs[ee]
                top4_jodis = [k for k, v in sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]]
                if pred['actual_jodi'] in top4_jodis:
                    daily_revenue += 4550
                    
            total_invested += daily_cost
            total_won += daily_revenue
            
        if total_invested > 0:
            roi = ((total_won - total_invested) / total_invested) * 100
            net = total_won - total_invested
            
            color_net = f"+₹{net:,}" if net > 0 else f"₹{net:,}"
            color_roi = f"+{roi:.2f}%" if roi > 0 else f"{roi:.2f}%"
            
            if roi > best_roi:
                best_roi = roi
                best_thresh = threshold
                
            print(f"> {threshold*100:.0f}% Conf | {days_played:<12} | ₹{total_invested:<9,} | ₹{total_won:<9,} | {color_net:<10} | {color_roi:<10}")
            
    print("\n=======================================================")
    if best_roi > 0:
        print(f"SUCCESS: Profitable Threshold Found at > {best_thresh*100:.0f}%")
        print(f"Maximized ROI: +{best_roi:.2f}%")
    else:
        print("CONCLUSION: Market is mathematically unbeatable even with filters.")
    print("=======================================================")

if __name__ == '__main__':
    main()
