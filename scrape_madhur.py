import re
import csv
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def parse_panna(th):
    text = th.get_text(strip=True).replace('*', '')
    if len(text) == 3 and text.isdigit():
        return text
    return ""

def parse_jodi(td):
    text = td.get_text(strip=True).replace('*', '')
    if len(text) == 2 and text.isdigit():
        return text
    return ""

html_file = r"C:\Users\admin\.gemini\antigravity-ide\brain\e1ca7c17-6610-4891-b927-4f6145ddf7de\.system_generated\steps\2436\content.md"

with open(html_file, 'r', encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

rows = soup.find_all('tr')

out_rows = []

for row in rows:
    cells = row.find_all(['th', 'td'])
    if not cells:
        continue
    
    # First cell is usually date range
    date_text = cells[0].get_text(strip=True)
    if 'to' not in date_text:
        continue
    
    parts = date_text.split('to')
    start_date_str = parts[0].strip()
    
    try:
        start_date = datetime.strptime(start_date_str, '%d/%m/%Y')
    except:
        try:
            start_date = datetime.strptime(start_date_str, '%d/%m/%y')
        except:
            continue
            
    # Process the 7 days (Mon to Sun)
    # Each day is 3 cells: TH (M Panna), TD (Jodi), TH (E Panna)
    # Index 1,2,3 -> Mon
    # Index 4,5,6 -> Tue
    
    for day_idx in range(7):
        base_idx = 1 + day_idx * 3
        if base_idx + 2 < len(cells):
            m_th = cells[base_idx]
            j_td = cells[base_idx + 1]
            e_th = cells[base_idx + 2]
            
            m_panna = parse_panna(m_th)
            jodi = parse_jodi(j_td)
            e_panna = parse_panna(e_th)
            
            if jodi:
                m_num = jodi[0]
                e_num = jodi[1]
                
                # If panna is missing but Jodi exists (sometimes happens in the chart), use '000'
                if not m_panna:
                    m_panna = '000'
                if not e_panna:
                    e_panna = '000'
                
                curr_date = start_date + timedelta(days=day_idx)
                
                out_rows.append({
                    'Date': curr_date.strftime('%Y-%m-%d'),
                    'Morning_panna': m_panna,
                    'Morning_number': m_num,
                    'Evening_number': e_num,
                    'Evening_panna': e_panna
                })

# Sort by date
out_rows.sort(key=lambda x: x['Date'])

with open('madhur_dataset.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Date', 'Morning_panna', 'Morning_number', 'Evening_number', 'Evening_panna'])
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Scraped {len(out_rows)} valid records to madhur_dataset.csv")
