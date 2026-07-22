import pandas as pd
import json
from collections import Counter

# Load Dates
with open('astrology/drikpanchang_dates.json', 'r', encoding='utf-8') as f:
    dp_data = json.load(f)

from datetime import datetime
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

def get_freqs(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Morning_number'] != '*') & (df['Evening_number'] != '*')]
    
    date_strs = list(all_dates_map.keys())
    
    df['is_astro'] = df['Date'].dt.strftime('%Y-%m-%d').isin(date_strs)
    
    astro_df = df[df['is_astro']]
    normal_df = df[~df['is_astro']]
    
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
    
    return m_diffs, e_diffs

madhur = pd.read_csv('data/madhur_dataset.csv')
rajdhani = pd.read_csv('data/rajdhani_day_dataset.csv')

r_m, r_e = get_freqs(rajdhani)
m_m, m_e = get_freqs(madhur)

print("=== Rajdhani Day ===")
print("Morning Difference (Astro % - Normal %):")
for k, v in sorted(r_m.items(), key=lambda x: x[1]):
    print(f"Digit {k}: {v:+.2f}%")
print("\nEvening Difference:")
for k, v in sorted(r_e.items(), key=lambda x: x[1]):
    print(f"Digit {k}: {v:+.2f}%")
    
print("\n=== Madhur ===")
print("Morning Difference:")
for k, v in sorted(m_m.items(), key=lambda x: x[1]):
    print(f"Digit {k}: {v:+.2f}%")
print("\nEvening Difference:")
for k, v in sorted(m_e.items(), key=lambda x: x[1]):
    print(f"Digit {k}: {v:+.2f}%")
