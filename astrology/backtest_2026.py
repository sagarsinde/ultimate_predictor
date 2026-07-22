import pandas as pd
import json
from collections import Counter
from datetime import datetime

# Load Dates
with open('astrology/drikpanchang_dates.json', 'r', encoding='utf-8') as f:
    dp_data = json.load(f)

def parse_dates(date_list, event_type):
    dt_list = []
    for d_str in date_list:
        parts = d_str.split(',')
        if len(parts) >= 2:
            clean_str = (parts[0] + ',' + parts[1]).strip()
            try:
                dt = datetime.strptime(clean_str, '%B %d, %Y')
                dt_list.append((dt, event_type))
            except ValueError:
                pass
    return dt_list

all_dates_map = {}
for year, data in dp_data.items():
    for dt, evt in parse_dates(data.get('Amavasya', []), 'Amavasya'):
        all_dates_map[dt.strftime('%Y-%m-%d')] = evt
    for dt, evt in parse_dates(data.get('Purnima', []), 'Purnima'):
        all_dates_map[dt.strftime('%Y-%m-%d')] = evt

def run_backtest(df, market_name):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Morning_number'] != '*') & (df['Evening_number'] != '*')]
    date_strs = list(all_dates_map.keys())
    df['is_astro'] = df['Date'].dt.strftime('%Y-%m-%d').isin(date_strs)
    
    # Split into Training (<= 2025) and Testing (2026)
    train_df = df[df['Date'].dt.year <= 2025]
    test_df = df[(df['Date'].dt.year == 2026) & df['is_astro']]
    
    # Calculate matrix on Training data
    astro_df = train_df[train_df['is_astro']]
    normal_df = train_df[~train_df['is_astro']]
    
    astro_m = Counter(astro_df['Morning_number'].astype(str).str.strip())
    astro_e = Counter(astro_df['Evening_number'].astype(str).str.strip())
    norm_m = Counter(normal_df['Morning_number'].astype(str).str.strip())
    norm_e = Counter(normal_df['Evening_number'].astype(str).str.strip())
    
    def calc_diff(astro, norm, total_astro, total_norm):
        diffs = {}
        for i in range(10):
            d = str(i)
            a_pct = (astro.get(d, 0) / total_astro) * 100 if total_astro else 0
            n_pct = (norm.get(d, 0) / total_norm) * 100 if total_norm else 0
            diffs[d] = a_pct - n_pct
        return diffs
        
    m_diffs = calc_diff(astro_m, norm_m, len(astro_df), len(normal_df))
    e_diffs = calc_diff(astro_e, norm_e, len(astro_df), len(normal_df))
    
    jodis = []
    for m in range(10):
        for e in range(10):
            score = m_diffs[str(m)] + e_diffs[str(e)]
            jodis.append({'Jodi': f"{m}{e}", 'Score': score})
            
    df_jodis = pd.DataFrame(jodis).sort_values('Score', ascending=False)
    top_10 = df_jodis.head(10)['Jodi'].tolist()
    
    # Now check 2026 results
    print(f"\n{'='*30}")
    print(f"BACKTEST: {market_name} 2026")
    print(f"{'='*30}")
    print(f"Model generated from {len(train_df)} past days.")
    print(f"Top 10 Predicted Jodis: {', '.join(top_10)}")
    print(f"--- 2026 Actual Astro Draws ---")
    
    hits = 0
    total_test_days = len(test_df)
    
    for _, row in test_df.sort_values('Date').iterrows():
        d_str = row['Date'].strftime('%Y-%m-%d')
        mn = str(row['Morning_number']).strip()
        en = str(row['Evening_number']).strip()
        jodi = f"{mn}{en}"
        evt = all_dates_map[d_str]
        
        hit_marker = "⭐⭐⭐ HIT!" if jodi in top_10 else ""
        if jodi in top_10:
            hits += 1
            
        print(f"{d_str} ({evt}): Drawn {jodi} {hit_marker}")
        
    print(f"\nResult: Out of {total_test_days} Astro days in 2026, the Top 10 predicted Jodis hit {hits} times.")
    
    # Random probability of hitting Top 10 out of 100 Jodis is 10%.
    expected_hits = total_test_days * 0.10
    print(f"Random expectation: {expected_hits:.1f} hits.")
    if hits > expected_hits:
        print("Model OUTPERFORMED random guessing.")
    else:
        print("Model underperformed or matched random guessing.")

madhur = pd.read_csv('data/madhur_dataset.csv')
rajdhani = pd.read_csv('data/rajdhani_day_dataset.csv')

run_backtest(rajdhani, 'Rajdhani Day')
run_backtest(madhur, 'Madhur')
