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

def get_diffs(df):
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

r_m, r_e = get_diffs(rajdhani)
m_m, m_e = get_diffs(madhur)

def generate_jodi_matrix(m_diff, e_diff, market_name):
    jodis = []
    for m in range(10):
        for e in range(10):
            score = m_diff[str(m)] + e_diff[str(e)]
            jodis.append({
                'Jodi': f"{m}{e}",
                'Morning_Digit': m,
                'Evening_Digit': e,
                'Morning_Boost': m_diff[str(m)],
                'Evening_Boost': e_diff[str(e)],
                'Total_Score': score
            })
            
    df_jodis = pd.DataFrame(jodis).sort_values('Total_Score', ascending=False)
    
    # Top 15 Jodis
    top_15 = df_jodis.head(15)
    
    # Bottom 15 Jodis (The Dead Jodis)
    bot_15 = df_jodis.tail(15).sort_values('Total_Score', ascending=True)
    
    out = f"## Market: {market_name}\n\n"
    out += "### Top 15 Highest Probability Jodis (Dealer Fallback)\n"
    out += "These are the mathematically safest Jodis for the dealer to draw on Astrological days, making them highly probable.\n\n"
    out += "| Jodi | Total Boost % | Morning Digit Boost | Evening Digit Boost |\n"
    out += "|---|---|---|---|\n"
    for _, row in top_15.iterrows():
        out += f"| **{row['Jodi']}** | +{row['Total_Score']:.2f}% | {row['Morning_Digit']} (+{row['Morning_Boost']:.2f}%) | {row['Evening_Digit']} (+{row['Evening_Boost']:.2f}%) |\n"
        
    out += "\n### Top 15 Dead Jodis (Gambler Fantasy Wipeout)\n"
    out += "These Jodis are massively suppressed by the dealer. Do not bet on these.\n\n"
    out += "| Jodi | Total Drop % | Morning Digit Drop | Evening Digit Drop |\n"
    out += "|---|---|---|---|\n"
    for _, row in bot_15.iterrows():
        out += f"| **{row['Jodi']}** | {row['Total_Score']:.2f}% | {row['Morning_Digit']} ({row['Morning_Boost']:.2f}%) | {row['Evening_Digit']} ({row['Evening_Boost']:.2f}%) |\n"
        
    out += "\n---\n\n"
    return out

report_str = "# Jodi Prediction Matrix for Amavasya & Purnima\n\n"
report_str += "This matrix cross-references the Morning and Evening digit manipulation rates to reveal the exact 2-digit Jodis that are heavily favored or completely suppressed by the dealer.\n\n"

report_str += generate_jodi_matrix(m_m, m_e, "Madhur")
report_str += generate_jodi_matrix(r_m, r_e, "Rajdhani Day")

with open('temp_jodi_report.md', 'w', encoding='utf-8') as f:
    f.write(report_str)
