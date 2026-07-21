import pandas as pd
import numpy as np
import joblib
import json

def main():
    print("--- Phase 5: Financial P&L Simulator (2026) ---")
    
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
    
    # 3. Unlock the 2026 Vault
    vault_df = df[df['Date'] >= '2026-01-01'].copy()
    print(f"Simulating Bank Account across {len(vault_df)} days in 2026...\n")
    
    # Financial Variables
    daily_cost = 500  # 150 (Morning) + 150 (Evening) + 200 (Jodi)
    payout_single = 450
    payout_jodi = 4550
    
    bankroll = 0
    total_invested = 0
    total_won = 0
    
    print("Format: [Date] | ACTUAL | M_PRED | E_PRED | JODI_PRED | DAILY_PROFIT | TOTAL_BANKROLL")
    print("-" * 110)
    
    # 4. Step through time
    for index, row in vault_df.iterrows():
        actual_m = int(row['Morning_number'])
        actual_e = int(row['Evening_number'])
        actual_jodi = f"{actual_m}{actual_e}"
        
        X_test = row[features].to_frame().T.astype(float)
        
        m_probs = model_m.predict_proba(X_test)[0]
        e_probs = model_e.predict_proba(X_test)[0]
        
        m_top3 = np.argsort(m_probs)[-3:][::-1].tolist()
        e_top3 = np.argsort(e_probs)[-3:][::-1].tolist()
        
        jodi_probs = {}
        for mm in range(10):
            for ee in range(10):
                jodi_probs[f"{mm}{ee}"] = m_probs[mm] * e_probs[ee]
                
        top4_jodis = [k for k, v in sorted(jodi_probs.items(), key=lambda x: x[1], reverse=True)[:4]]
        
        # Financial Resolution for the Day
        daily_revenue = 0
        
        if actual_m in m_top3:
            daily_revenue += payout_single
        if actual_e in e_top3:
            daily_revenue += payout_single
        if actual_jodi in top4_jodis:
            daily_revenue += payout_jodi
            
        daily_profit = daily_revenue - daily_cost
        bankroll += daily_profit
        
        total_invested += daily_cost
        total_won += daily_revenue
        
        date_str = row['Date'].strftime('%Y-%m-%d')
        
        # Color coding for terminal (Optional, but nice if run locally)
        profit_str = f"+{daily_profit}" if daily_profit > 0 else str(daily_profit)
        
        print(f"[{date_str}] | {actual_jodi} | {m_top3} | {e_top3} | {top4_jodis} | {profit_str:^6} | {bankroll}")
        
    # 5. Print the Final Scorecard
    print("\n=======================================================")
    print("        2026 FINANCIAL P&L SCORECARD")
    print("=======================================================")
    print(f"Total Days Played:  {len(vault_df)}")
    print(f"Total Invested:    ₹{total_invested:,}")
    print(f"Total Won:         ₹{total_won:,}")
    print("-" * 55)
    
    if bankroll > 0:
        print(f"NET PROFIT:       +₹{bankroll:,}  (WINNING STRATEGY)")
    else:
        print(f"NET LOSS:          ₹{bankroll:,}  (LOSING STRATEGY)")
        
    roi = (total_won - total_invested) / total_invested * 100
    print(f"Total ROI:         {roi:+.2f}%")
    print("=======================================================\n")

if __name__ == '__main__':
    main()
