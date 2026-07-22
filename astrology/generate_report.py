import pandas as pd
import json
import os
from datetime import datetime

# Load JSON
with open('astrology/drikpanchang_dates.json', 'r', encoding='utf-8') as f:
    dp_data = json.load(f)

def parse_dates(date_list):
    dt_list = []
    for d_str in date_list:
        parts = d_str.split(',')
        if len(parts) >= 2:
            clean_str = (parts[0] + ',' + parts[1]).strip()
            try:
                dt = datetime.strptime(clean_str, '%B %d, %Y')
                dt_list.append(dt)
            except ValueError:
                pass
    return dt_list

all_ama_dates = []
all_pur_dates = []

for year, data in dp_data.items():
    all_ama_dates.extend(parse_dates(data.get('Amavasya', [])))
    all_pur_dates.extend(parse_dates(data.get('Purnima', [])))

def analyze_market(df, market_name):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Morning_number'] != '*') & (df['Evening_number'] != '*')]
    
    ama_df = df[df['Date'].isin(all_ama_dates)]
    pur_df = df[df['Date'].isin(all_pur_dates)]
    
    def get_missing_by_year(subset_df):
        out = []
        if len(subset_df) == 0:
            return out
        subset_df['Year'] = subset_df['Date'].dt.year
        for y in sorted(subset_df['Year'].unique()):
            y_df = subset_df[subset_df['Year'] == y]
            m_nums = set(y_df['Morning_number'].astype(str).str.strip())
            e_nums = set(y_df['Evening_number'].astype(str).str.strip())
            
            all_digits = set([str(i) for i in range(10)])
            m_miss = sorted(list(all_digits - m_nums))
            e_miss = sorted(list(all_digits - e_nums))
            out.append((y, len(y_df), m_miss, e_miss))
        return out
        
    return {
        'Amavasya': get_missing_by_year(ama_df),
        'Purnima': get_missing_by_year(pur_df)
    }

madhur = pd.read_csv('data/madhur_dataset.csv')
rajdhani = pd.read_csv('data/rajdhani_day_dataset.csv')

results = {
    'Madhur': analyze_market(madhur, 'Madhur'),
    'Rajdhani Day': analyze_market(rajdhani, 'Rajdhani Day')
}

with open('temp_report.md', 'w') as f:
    f.write('# DrikPanchang Astrological Manipulation Report (2013 - 2026)\n\n')
    f.write('This report analyzes the missing numbers for Amavasya and Purnima using exact dates scraped directly from DrikPanchang\'s archive.\n\n')
    
    for market, m_data in results.items():
        f.write(f'## Market: {market}\n\n')
        
        for event in ['Amavasya', 'Purnima']:
            f.write(f'### {event}\n')
            f.write('| Year | Days Count | Morning Missing | Evening Missing |\n')
            f.write('|---|---|---|---|\n')
            for row in m_data[event]:
                y, count, m_miss, e_miss = row
                m_str = ','.join(m_miss) if m_miss else 'None'
                e_str = ','.join(e_miss) if e_miss else 'None'
                f.write(f'| {y} | {count} | `{m_str}` | `{e_str}` |\n')
            f.write('\n')
