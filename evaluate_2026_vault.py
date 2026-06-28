import pandas as pd
import numpy as np
import joblib
import json

def main():
    print("--- Phase 4: The 2026 Vault Evaluator ---")
    
    # 1. Load the Brains
    try:
        model_m = joblib.load('morning_model.joblib')
        model_e = joblib.load('evening_model.joblib')
        with open('model_features.json', 'r') as f:
            features = json.load(f)
    except FileNotFoundError:
        print("Error: Models not found. Run train_model.py in Colab first.")
        return
        
    # 2. Load the Feature Store
    print("Opening the Data Lake...")
    df = pd.read_csv('kalyan_feature_store.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 3. Unlock the 2026 Vault (The Hold-Out Set)
    vault_df = df[df['Date'] >= '2026-01-01'].copy()
    print(f"2026 Vault Unlocked. Testing AI on {len(vault_df)} unseen draws...\n")
    
    # Scorecard Tracking
    m_top1_hits = 0
    m_top3_hits = 0
    
    e_top1_hits = 0
    e_top3_hits = 0
    
    jodi_top4_hits = 0
    total_draws = len(vault_df)
    
    if total_draws == 0:
        print("Error: No data found for 2026 in the feature store.")
        return

    print("Simulating 2026 Timeline...")
    print("Format: [Date] | ACTUAL: [Morning] [Evening] | AI MORNING TOP 3 | AI EVENING TOP 3")
    print("-" * 90)
    
    # 4. Step through time
    for index, row in vault_df.iterrows():
        # Get actual results
        actual_m = int(row['Morning_number'])
        actual_e = int(row['Evening_number'])
        actual_jodi = f"{actual_m}{actual_e}"
        
        # Extract features for prediction
        X_test = row[features].to_frame().T.astype(float)
        
        # Predict Probas
        m_probs = model_m.predict_proba(X_test)[0]
        e_probs = model_e.predict_proba(X_test)[0]
        
        # Get Top Picks
        m_top1 = np.argsort(m_probs)[-1]
        m_top3 = np.argsort(m_probs)[-3:][::-1].tolist()
        
        e_top1 = np.argsort(e_probs)[-1]
        e_top3 = np.argsort(e_probs)[-3:][::-1].tolist()
        
        date_str = row['Date'].strftime('%Y-%m-%d')
        print(f"[{date_str}] | ACTUAL: {actual_m} {actual_e} | AI M_PRED: {m_top3} | AI E_PRED: {e_top3}")
        
        # Joint Jodi Probabilities
        jodi_probs = {}
        for mm in range(10):
            for ee in range(10):
                jodi_probs[f"{mm}{ee}"] = m_probs[mm] * e_probs[ee]
                
        # Get Top 4 Jodis
        top4_jodis = [k for k, v in sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]]
        
        # Grade the AI
        if actual_m == m_top1: m_top1_hits += 1
        if actual_m in m_top3: m_top3_hits += 1
        
        if actual_e == e_top1: e_top1_hits += 1
        if actual_e in e_top3: e_top3_hits += 1
        
        if actual_jodi in top4_jodis: jodi_top4_hits += 1
        
    # 5. Print the Final Scorecard
    print("\n=======================================================")
    print("      ADVANCED LIE-DETECTOR SCORECARD (2026 VAULT)")
    print("=======================================================")
    print(f"Total Test Period Draws Evaluated: {total_draws}\n")
    
    print("--- MORNING MODEL ---")
    print(f"Top 1 Hit Rate: {(m_top1_hits/total_draws)*100:.2f}%  (Random Baseline: 10%)")
    print(f"Top 3 Hit Rate: {(m_top3_hits/total_draws)*100:.2f}%  (Random Baseline: 30%)")
    
    print("\n--- EVENING MODEL ---")
    print(f"Top 1 Hit Rate: {(e_top1_hits/total_draws)*100:.2f}%  (Random Baseline: 10%)")
    print(f"Top 3 Hit Rate: {(e_top3_hits/total_draws)*100:.2f}%  (Random Baseline: 30%)")
    
    print("\n--- JODI JOINT PROBABILITY ---")
    print(f"Top 4 Jodi Hit Rate: {(jodi_top4_hits/total_draws)*100:.2f}%  (Random Baseline: 4%)")
    print("=======================================================\n")
    print("If Hit Rates significantly beat Baseline, the dealer is acting predictably!")

if __name__ == '__main__':
    main()
