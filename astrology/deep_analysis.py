import pandas as pd
import json
import os
from datetime import datetime

# Load JSON
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

def generate_deep_report(df, market_name):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Morning_number'] != '*') & (df['Evening_number'] != '*')]
    
    # Filter only the exact dates
    date_strs = list(all_dates_map.keys())
    df_filtered = df[df['Date'].dt.strftime('%Y-%m-%d').isin(date_strs)].copy()
    
    if len(df_filtered) == 0:
        return ""
        
    df_filtered['Event'] = df_filtered['Date'].dt.strftime('%Y-%m-%d').map(all_dates_map)
    df_filtered['Year'] = df_filtered['Date'].dt.year
    df_filtered = df_filtered.sort_values('Date')
    
    out = f"## Market: {market_name}\n\n"
    for year in sorted(df_filtered['Year'].unique()):
        out += f"### Year {year}\n"
        out += "| Date | Event | Morning Pana | Morning | Evening | Evening Pana |\n"
        out += "|---|---|---|---|---|---|\n"
        
        year_df = df_filtered[df_filtered['Year'] == year]
        for _, row in year_df.iterrows():
            d_str = row['Date'].strftime('%Y-%m-%d')
            evt = row['Event']
            
            if 'Morning_Panna' in row:
                mp = str(row['Morning_Panna']).strip()
                ep = str(row['Evening_Panna']).strip()
            else:
                mp = str(row['Morning_pana']).strip()
                ep = str(row['Evening_pana']).strip()
                
            mn = str(row['Morning_number']).strip()
            en = str(row['Evening_number']).strip()
            
            # Format nicely
            out += f"| {d_str} | **{evt}** | `{mp}` | **{mn}** | **{en}** | `{ep}` |\n"
        out += "\n"
        
    return out

madhur = pd.read_csv('data/madhur_dataset.csv')
rajdhani = pd.read_csv('data/rajdhani_day_dataset.csv')

report_str = "# Deep Analysis: Actual Draws on Amavasya & Purnima\n\n"
report_str += "This report lists every single exact draw result for the correct DrikPanchang dates so we can spot micro-patterns.\n\n"

report_str += generate_deep_report(rajdhani, 'Rajdhani Day')
report_str += generate_deep_report(madhur, 'Madhur')

with open('temp_deep_report.md', 'w', encoding='utf-8') as f:
    f.write(report_str)
