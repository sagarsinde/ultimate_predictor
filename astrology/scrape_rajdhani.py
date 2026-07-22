import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta

def parse_date(date_str):
    try:
        return datetime.strptime(date_str.strip(), '%d/%m/%y')
    except:
        return None

def parse_rajdhani_html(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()
        
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table', class_='record')
    
    rows_data = []
    
    for table in tables:
        for tr in table.find_all('tr'):
            tds_ths = tr.find_all(['td', 'th'])
            if not tds_ths or len(tds_ths) < 18:
                continue
                
            week_td = tds_ths[0].text
            if 'to' not in week_td:
                continue
                
            week_parts = week_td.split('to')
            if len(week_parts) != 2:
                continue
                
            start_date_str = week_parts[0].strip()
            start_date = parse_date(start_date_str)
            if not start_date:
                continue
            
            # The structure for Monday to Saturday (6 days)
            # td0 = Date
            # Mon: td1(MPana), th2(Jodi), td3(EPana)
            # Tue: td4, th5, td6
            # Wed: td7, th8, td9
            # Thu: td10, th11, td12
            # Fri: td13, th14, td15
            # Sat: td16, th17, td18
            
            day_idx = 0
            for i in range(1, 19, 3):
                if i + 2 >= len(tds_ths):
                    break
                    
                m_pana = tds_ths[i].text.strip().replace('\n', '').replace(' ', '')
                jodi = tds_ths[i+1].text.strip()
                e_pana = tds_ths[i+2].text.strip().replace('\n', '').replace(' ', '')
                
                # Check for holiday / red stars
                if m_pana == '***' or m_pana == '**' or not m_pana or '*' in m_pana:
                    m_pana = '*'
                if e_pana == '***' or e_pana == '**' or not e_pana or '*' in e_pana:
                    e_pana = '*'
                
                if jodi == '**' or not jodi or '*' in jodi:
                    m_num = '*'
                    e_num = '*'
                else:
                    jodi_clean = re.sub(r'[^0-9]', '', jodi)
                    if len(jodi_clean) == 2:
                        m_num = jodi_clean[0]
                        e_num = jodi_clean[1]
                    else:
                        m_num = '*'
                        e_num = '*'
                        
                current_date = start_date + timedelta(days=day_idx)
                date_formatted = current_date.strftime('%Y-%m-%d')
                
                rows_data.append({
                    'Date': date_formatted,
                    'Morning_pana': m_pana,
                    'Morning_number': m_num,
                    'Evening_number': e_num,
                    'Evening_pana': e_pana
                })
                
                day_idx += 1
                
    df = pd.DataFrame(rows_data)
    return df

if __name__ == '__main__':
    print('Downloading HTML...')
    resp = requests.get('https://sattamatkajico.com/panna/rajdhani-day-chart-record.php', headers={'User-Agent': 'Mozilla/5.0'})
    with open('rajdhani_temp.html', 'w', encoding='utf-8') as f:
        f.write(resp.text)
        
    print('Parsing HTML...')
    df_new = parse_rajdhani_html('rajdhani_temp.html')
    
    # Check if we have data
    if len(df_new) > 0:
        # Load existing
        df_existing = pd.read_csv('data/rajdhani_day_dataset.csv')
        df_existing['Date'] = pd.to_datetime(df_existing['Date']).dt.strftime('%Y-%m-%d')
        
        # Combine
        # The new dataset from sattamatkajico has data up to present.
        # But we only want to append the older data that we don't have.
        # Let's just merge them and drop duplicates based on Date, keeping the existing one where possible, or taking new one.
        df_combined = pd.concat([df_new, df_existing]).drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
        
        df_combined.to_csv('data/rajdhani_day_dataset.csv', index=False)
        print(f'Successfully combined datasets. Total rows: {len(df_combined)}')
        print(f'Oldest date: {df_combined["Date"].min()}')
        print(f'Newest date: {df_combined["Date"].max()}')
    else:
        print('No data parsed!')
